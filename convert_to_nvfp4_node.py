import os
import json
import torch
import folder_paths
import safetensors.torch
import comfy.utils
import safetensors
from collections import OrderedDict

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout, TensorWiseINT8Layout, TensorCoreConvRotW4A4Layout
    try:
        from comfy_kitchen.tensor import TensorCoreMXFP8Layout
        MXFP8_AVAILABLE = True
    except ImportError:
        MXFP8_AVAILABLE = False
        print("⚠️ [Convert-to-NVFP4] MXFP8 indisponible, mettez comfy-kitchen à jour.")
    try:
        from comfy_kitchen.tensor import AsymW4A8Int8Layout
        W4A8_AVAILABLE = True
    except ImportError:
        W4A8_AVAILABLE = False
        print("⚠️ [Convert-to-NVFP4] W4A8 indisponible, mettez comfy-kitchen à jour.")
except ImportError:
    print("⚠️ [Convert-to-NVFP4] comfy-kitchen introuvable.")
    MXFP8_AVAILABLE = False
    W4A8_AVAILABLE = False

CONVROT_GROUPSIZE = 256
INT4_QUANT_GROUPSIZE = 64
W4A8_GROUPSIZE = 16


def _canonical_layer_name(name):
    for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _read_source_quantization_configs(input_path, state_dict):
    configs = {}

    def add_config(layer_name, config):
        if not isinstance(config, dict):
            raise ValueError(f"Invalid quantization config for layer {layer_name}")
        canonical_name = _canonical_layer_name(layer_name)
        previous = configs.get(canonical_name)
        if previous is not None and previous != config:
            raise ValueError(f"Conflicting quantization configs for layer {layer_name}")
        configs[canonical_name] = config

    for key, tensor in state_dict.items():
        if not key.endswith(".comfy_quant"):
            continue
        raw_config = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
        add_config(key[:-len(".comfy_quant")], json.loads(raw_config.decode("utf-8")))

    with safetensors.safe_open(input_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    metadata_quantization = metadata.get("_quantization_metadata")
    if metadata_quantization:
        quantization = json.loads(metadata_quantization)
        for layer_name, config in quantization.get("layers", {}).items():
            add_config(layer_name, config)

    return configs


def _dequantize_int8_weight(qdata, scale, layer_config, device):
    params_config = layer_config.get("params", {})
    if not isinstance(params_config, dict):
        params_config = {}
    convrot = bool(layer_config.get("convrot", params_config.get("convrot", False)))
    convrot_groupsize = int(
        layer_config.get("convrot_groupsize", params_config.get("convrot_groupsize", CONVROT_GROUPSIZE))
    )

    if qdata.dtype == torch.uint8:
        qdata = qdata.view(torch.int8)
    elif qdata.dtype != torch.int8:
        raise ValueError(f"Expected INT8 weight data, got {qdata.dtype}")

    params = TensorWiseINT8Layout.Params(
        scale=scale.to(device=device, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qdata.shape),
        is_weight=True,
        convrot=convrot,
        convrot_groupsize=convrot_groupsize,
    )
    return TensorWiseINT8Layout.dequantize(qdata.to(device=device), params)


def _pick_convrot_groupsize(k, preferred=CONVROT_GROUPSIZE):
    """Plus grand groupsize ConvRot divisant K.

    Le layout W4A8 exige K % convrot_groupsize == 0. Toutes les couches
    ne sont pas divisibles par 256 (ex. K=2688 sur MiniMax H3), on
    redescend donc par paliers plutot que d'echouer.
    """
    for gs in (preferred, 128, 64, 32, 16):
        if gs <= k and k % gs == 0:
            return gs
    return None

class ConvertToNVFP4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "output_filename": ("STRING", {"default": "",
                    "placeholder": "vide = nom du modele source",
                    "tooltip": "Le suffixe du format choisi est ajoute automatiquement."}),
                "model_type": ([
                    "Z-Image-Turbo", 
                    "Z-Image-Base", 
                    "Flux.1-dev", 
                    "Flux.1-Fill", 
                    "Flux.2-dev", 
                    "Flux.2-Klein-9b",
                    "Qwen-Image-Edit-2511", 
                    "Qwen-Image-2512", 
                    "Wan2.2-i2v-high-low",
                    "LTX-2-19b-dev-or-distilled",
                    "Krea2",
                    "ACE-Step",
                    "Anima",
                    "Boogu-Image",
                    "Chroma",
                    "ERNIE-Image",
                    "Ideogram-4",
                    "MiniMax-H3",
                    "SeedVR"
                ], {"default": "Z-Image-Turbo"}),
                "quant_format": (["NVFP4", "MXFP8", "INT8_CONVROT", "INT4_CONVROT", "W4A8_INT8"], {"default": "NVFP4"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "Kitchen"
    OUTPUT_NODE = True

    def convert(self, model_name, output_filename, model_type, quant_format, device):
        input_path = folder_paths.get_full_path("diffusion_models", model_name)
        # Nom de sortie : base choisie (ou nom du modele source) + suffixe du format.
        suffix = {"NVFP4": "nvfp4", "MXFP8": "mxfp8",
                  "INT8_CONVROT": "int8_convrot",
                  "INT4_CONVROT": "int4_convrot"}.get(quant_format, quant_format.lower())
        base = (output_filename or "").strip()
        if not base:
            base = os.path.splitext(os.path.basename(model_name))[0]
        if base.lower().endswith(".safetensors"):
            base = base[: -len(".safetensors")]
        final_name = f"{base}_{suffix}"
        output_path = os.path.join(os.path.dirname(input_path), f"{final_name}.safetensors")
        
        # --- CONFIGURATION DES PROFILS ---
        if model_type == "Qwen-Image-Edit-2511":
            BLACKLIST = ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"]
            FP8_LAYERS = []
        elif model_type == "Qwen-Image-2512":
            BLACKLIST = ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out", "img_mod.1"]
            FP8_LAYERS = ["txt_mlp", "txt_mod"]
        elif model_type == "Wan2.2-i2v-high-low":
            BLACKLIST = ["text_embedding", "time_embedding", "time_projection", "head"]
            FP8_LAYERS = []
        elif model_type in ["Flux.1-dev", "Flux.1-Fill", "Flux.2-dev", "Flux.2-Klein-9b"]:
            BLACKLIST = ["bias", "txt_attn", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer", "class_embedding", "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt"]
            FP8_LAYERS = []
        elif model_type == "Z-Image-Base":
            BLACKLIST = ["attention", "adaLN_modulation", "norm", "final_layer", "cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder"]
            FP8_LAYERS = []
        elif model_type == "Z-Image-Turbo":
            BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer"]
            FP8_LAYERS = []
        elif model_type == "LTX-2-19b-dev-or-distilled":
            BLACKLIST = [
                "vae.", "vocoder.", "connector", "proj_out",
                "norm", "bias", "scale", "embedder", "patchify", "table",
                "transformer_blocks.0.",
                "transformer_blocks.43.", "transformer_blocks.44.", 
                "transformer_blocks.45.", "transformer_blocks.46.", 
                "transformer_blocks.47.", "projection", "adaln_single"
            ]
            FP8_LAYERS = []
        elif model_type == "ACE-Step":
            BLACKLIST = ["bias", "norm", "scale", "final_layer", "proj_out", "head", "text_embedding", "time_embedding", "time_projection", "embedder", "adaln", "t_embedder", "x_embedder", "y_embedder", "project_in", "quantizer", "embed_tokens", "timbre_encoder"]
            FP8_LAYERS = []
        elif model_type == "Anima":
            BLACKLIST = ["bias", "norm", "scale", "llm_adapter", "final_layer", "proj_out", "x_embedder", "t_embedder", "context_embedder"]
            FP8_LAYERS = []
        elif model_type == "Boogu-Image":
            BLACKLIST = ["image_index_embedding", "ref_image_patch_embedder", "norm1.linear", "norm_out.linear_1", "norm_out.linear_2"]
            FP8_LAYERS = []
        elif model_type == "Chroma":
            BLACKLIST = ["bias", "txt_attn", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer", "class_embedding", "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt"]
            FP8_LAYERS = []
        elif model_type == "ERNIE-Image":
            BLACKLIST = ["bias", "norm", "scale", "final_layer", "proj_out", "head", "text_embedding", "time_embedding", "time_projection", "embedder", "adaln", "t_embedder", "x_embedder", "y_embedder", "context_embedder", "single_stream_modulation", "enhancer"]
            FP8_LAYERS = []
        elif model_type == "Ideogram-4":
            BLACKLIST = ["bias", "norm", "scale", "final_layer", "proj_out", "x_embedder", "t_embedder", "context_embedder", "text_embedding", "time_embedding", "modulation", "adaLN", "q_norm", "k_norm", "img_in", "txt_in", "time_in", "vector_in", "guidance_in", "single_stream_modulation", "embed_image_indicator", "embed_text_indicator", "adaln_proj", "input_proj", "llm_cond_proj", "t_embedding"]
            FP8_LAYERS = []
        elif model_type == "MiniMax-H3":
            BLACKLIST = ["bias", "norm", "adaln_proj", "adaln_t_table", "condition_proj", "final_layer", "token_refiner", "patch_proj", "rope"]
            FP8_LAYERS = []
        elif model_type == "SeedVR":
            BLACKLIST = ["bias", "norm", "scale", "final_layer", "proj_out", "pos_emb", "neg_emb", "patch_embed", "pos_embed", "x_embedder", "t_embedder", "context_embedder", "y_embedder", "time_in", "vector_in", "guidance_in", "txt_in", "img_in", "single_stream_modulation", "double_stream_modulation", "vae", "attn.proj_qkv_vid"]
            FP8_LAYERS = []
        elif model_type == "Krea2":
            BLACKLIST = ["first", "last", "tmlp", "tproj", "txtfusion", "txtmlp"]
            FP8_LAYERS = []
        else:
            BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer"]
            FP8_LAYERS = []

        print(f"🚀 Mode {model_type} | Format {quant_format}")
        
        temp_diffusers_meta = {}
        if model_type == "LTX-2-19b-dev-or-distilled":
            with safetensors.safe_open(input_path, framework="pt") as f:
                orig_meta = f.metadata()
                if orig_meta:
                    for key in ["config", "license", "encrypted_wandb_properties"]:
                        if key in orig_meta:
                            temp_diffusers_meta[key] = orig_meta[key]

        sd = safetensors.torch.load_file(input_path)
        source_quant_configs = _read_source_quantization_configs(input_path, sd)
        source_quant_configs_by_weight = {}
        source_quant_scale_keys = set()
        matched_source_quant_layers = set()
        for key, tensor in sd.items():
            if not key.endswith(".weight"):
                continue
            layer_name = key[:-len(".weight")]
            canonical_name = _canonical_layer_name(layer_name)
            layer_config = source_quant_configs.get(canonical_name)
            if layer_config is None:
                continue
            if tensor.ndim != 2:
                raise ValueError(f"Quantized source layer {layer_name} is not a 2D weight")
            if layer_config.get("format") != "int8_tensorwise":
                raise ValueError(
                    f"Cannot convert source quantization format "
                    f"{layer_config.get('format')} for layer {layer_name}; "
                    "only INT8 ConvRot is supported as a quantized input."
                )
            params_config = layer_config.get("params", {})
            if not isinstance(params_config, dict):
                params_config = {}
            if not layer_config.get("convrot", params_config.get("convrot", False)):
                raise ValueError(f"INT8 layer {layer_name} is not ConvRot-quantized")
            source_quant_configs_by_weight[key] = layer_config
            source_quant_scale_keys.add(f"{layer_name}.weight_scale")
            matched_source_quant_layers.add(canonical_name)

        unmatched_source_quant_layers = set(source_quant_configs) - matched_source_quant_layers
        if unmatched_source_quant_layers:
            first_layer = sorted(unmatched_source_quant_layers)[0]
            raise ValueError(f"Quantized source metadata has no matching weight for layer {first_layer}")

        quant_map = {"format_version": "1.0", "layers": {}}
        new_sd = {}
        
        pbar = comfy.utils.ProgressBar(len(sd))
        print(f"⚙️ Conversion lancée sur : {device}")

        for i, (k, v) in enumerate(sd.items()):
            pbar.update_absolute(i + 1)

            if k.endswith(".comfy_quant") or k in source_quant_scale_keys:
                continue

            if v.ndim == 2 and ".weight" in k:
                base_k_file = k.replace(".weight", "")
                source_layer_config = source_quant_configs_by_weight.get(k)
                scale = None
                if source_layer_config is not None:
                    scale = sd.get(f"{base_k_file}.weight_scale")
                    if scale is None:
                        raise ValueError(f"Missing INT8 weight scale for layer {base_k_file}")

                if any(name in k for name in BLACKLIST):
                    if source_layer_config is None:
                        new_sd[k] = v.to(dtype=torch.bfloat16)
                    else:
                        v_tensor = _dequantize_int8_weight(v, scale, source_layer_config, device)
                        new_sd[k] = v_tensor.to(dtype=torch.bfloat16).cpu()
                        if device == "cuda": del v_tensor
                    continue

                if source_layer_config is not None:
                    v_tensor = _dequantize_int8_weight(v, scale, source_layer_config, device)
                else:
                    v_tensor = v.to(device=device, dtype=torch.bfloat16)

                if model_type == "LTX-2-19b-dev-or-distilled":
                    base_k_meta = k.replace(".weight", "") 
                else:
                    if "model.diffusion_model." in base_k_file:
                        base_k_meta = base_k_file.split("model.diffusion_model.")[-1]
                    else:
                        base_k_meta = base_k_file
      
                if FP8_LAYERS and any(name in k for name in FP8_LAYERS):
                    print(f"🌸 FP8 Cuisine : {k}")
                    weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                    weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                    new_sd[k] = weight_quantized.cpu()
                    new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                    quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                    if device == "cuda": del v_tensor
                    continue

            elif any(name in k for name in BLACKLIST):
                new_sd[k] = v.to(dtype=torch.bfloat16)
                continue
            else:
                new_sd[k] = v.to(dtype=torch.bfloat16)
                continue

            if v.ndim == 2 and ".weight" in k:
                print(f"💎 {quant_format} : {k}")
                try:
                    # Selection du layout et des arguments propres a chaque format
                    if quant_format == "INT8_CONVROT":
                        layout = TensorWiseINT8Layout
                        qdata, params = layout.quantize(
                            v_tensor, is_weight=True, per_channel=True,
                            convrot=True, convrot_groupsize=CONVROT_GROUPSIZE,
                            stochastic_rounding=0)
                        layer_conf = {"format": "int8_tensorwise", "convrot": True,
                                      "convrot_groupsize": CONVROT_GROUPSIZE}
                    elif quant_format == "INT4_CONVROT":
                        layout = TensorCoreConvRotW4A4Layout
                        qdata, params = layout.quantize(
                            v_tensor, convrot_groupsize=CONVROT_GROUPSIZE,
                            quant_group_size=INT4_QUANT_GROUPSIZE,
                            stochastic_rounding=0, linear_dtype="int4")
                        layer_conf = {"format": "convrot_w4a4",
                                      "convrot_groupsize": CONVROT_GROUPSIZE,
                                      "quant_group_size": INT4_QUANT_GROUPSIZE,
                                      "linear_dtype": "int4"}
                    elif quant_format == "W4A8_INT8":
                        if not W4A8_AVAILABLE:
                            raise RuntimeError("W4A8 indisponible dans cette version de comfy-kitchen")
                        # K doit etre divisible par convrot_groupsize : on prend
                        # le plus grand palier valide pour cette couche.
                        k_dim = v_tensor.shape[1]
                        cgs = _pick_convrot_groupsize(k_dim)
                        if cgs is None or k_dim % 16 != 0:
                            raise ValueError(f"K={k_dim} incompatible W4A8 (doit etre multiple de 16)")
                        layout = AsymW4A8Int8Layout
                        qdata, params = layout.quantize(
                            v_tensor, group_size=W4A8_GROUPSIZE,
                            convrot_groupsize=cgs, symmetric=True,
                            codebook=True, stochastic_rounding=0)
                        layer_conf = {"format": "asym_w4a8_int8",
                                      "group_size": W4A8_GROUPSIZE,
                                      "convrot_groupsize": cgs}
                    elif quant_format == "MXFP8":
                        if not MXFP8_AVAILABLE:
                            raise RuntimeError("MXFP8 indisponible dans cette version de comfy-kitchen")
                        layout = TensorCoreMXFP8Layout
                        qdata, params = layout.quantize(v_tensor)
                        layer_conf = {"format": "mxfp8"}
                    else:
                        layout = TensorCoreNVFP4Layout
                        qdata, params = layout.quantize(v_tensor)
                        layer_conf = {"format": "nvfp4"}

                    tensors = layout.state_dict_tensors(qdata, params)
                    for suffix, tensor in tensors.items():
                        # Certains scales sortent en dtypes que safetensors ne sait
                        # pas serialiser : on les reinterprete en uint8.
                        if tensor.dtype == torch.float8_e8m0fnu:
                            new_sd[f"{base_k_file}.weight{suffix}"] = tensor.view(torch.uint8).cpu()
                        elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                            # passage par uint8 pour garantir la contiguite
                            new_sd[f"{base_k_file}.weight{suffix}"] = tensor.view(torch.uint8).cpu().view(tensor.dtype)
                        else:
                            new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                    quant_map["layers"][base_k_meta] = layer_conf
                except Exception as e:
                    print(f"⚠️ {k} non quantifiable ({e}) → bf16")
                    new_sd[k] = v_tensor.to(dtype=torch.bfloat16).cpu()
                
                if device == "cuda": del v_tensor

        final_metadata = OrderedDict()
        final_metadata["_quantization_metadata"] = json.dumps(quant_map)
        
        final_metadata["converted_by"] = "ComfyUI Kitchen NVFP4 Converter"
        final_metadata["quant_format"] = quant_format
        final_metadata["converter_url"] = "https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter"
        
        if model_type == "LTX-2-19b-dev-or-distilled":
            for k, v in temp_diffusers_meta.items():
                final_metadata[k] = v

        print(f"💾 Saving file | Type: {model_type} | Path: {output_path}")
        safetensors.torch.save_file(new_sd, output_path, metadata=final_metadata)
        
        total_bytes = os.path.getsize(output_path)
        print(f"✅ Terminé. Taille finale : {round(total_bytes / (1024**3), 2)} Go")
        
        return (f"Succès ({model_type} / {quant_format}) : {final_name}.safetensors",)

NODE_CLASS_MAPPINGS = {"ConvertToNVFP4": ConvertToNVFP4}
NODE_DISPLAY_NAME_MAPPINGS = {"ConvertToNVFP4": "🍳 Kitchen Quant Converter"}