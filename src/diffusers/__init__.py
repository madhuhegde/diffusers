__version__ = "0.24.0.dev0"

from typing import TYPE_CHECKING

from .utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_onnx_available,
    is_scipy_available,
    is_torch_available,
    is_torchsde_available,
    is_transformers_available,
)


# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "models": [],
    "pipelines": [],
    "schedulers": [],
    "utils": [
        "OptionalDependencyNotAvailable",
        "is_flax_available",
        "is_inflect_available",
        "is_invisible_watermark_available",
        "is_k_diffusion_available",
        "is_k_diffusion_version",
        "is_librosa_available",
        "is_note_seq_available",
        "is_onnx_available",
        "is_scipy_available",
        "is_torch_available",
        "is_torchsde_available",
        "is_transformers_available",
        "is_transformers_version",
        "is_unidecode_available",
        "logging",
    ],
}
_import_structure["pipelines"].extend("LDMSuperResolutionPipeline")
from .pipelines import LDMSuperResolutionPipeline
try:
    if not is_onnx_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_onnx_objects  # noqa F403

    _import_structure["utils.dummy_onnx_objects"] = [
        name for name in dir(dummy_onnx_objects) if not name.startswith("_")
    ]

else:
    _import_structure["pipelines"].extend(["OnnxRuntimeModel"])

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .utils import dummy_pt_objects  # noqa F403

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]

else:
    _import_structure["models"].extend(
        [
            #"AsymmetricAutoencoderKL",
            #"AutoencoderKL",
            #"AutoencoderTiny",
            #"ConsistencyDecoderVAE",
            #"ControlNetModel",
            #"Kandinsky3UNet",
            "ModelMixin",
            #"MotionAdapter",
            #"MultiAdapter",
            #"PriorTransformer",
            #"T2IAdapter",
            #"T5FilmDecoder",
            #"Transformer2DModel",
            #"UNet1DModel",
            #"UNet2DConditionModel",
            "UNet2DModel",
            #"UNet3DConditionModel",
            #"UNetMotionModel",
            "VQModel",
        ]
    )

    _import_structure["optimization"] = [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
    _import_structure["pipelines"].extend(
        [
            #"AudioPipelineOutput",
            #"AutoPipelineForImage2Image",
            #"AutoPipelineForInpainting",
            #"AutoPipelineForText2Image",
            #"ConsistencyModelPipeline",
            #"DanceDiffusionPipeline",
            #"DDIMPipeline",
            #"DDPMPipeline",
            "DiffusionPipeline",
            #"DiTPipeline",
            #"ImagePipelineOutput",
            #"KarrasVePipeline",
            #"LDMPipeline",
            "LDMSuperResolutionPipeline",
            #"PNDMPipeline",
            #"RePaintPipeline",
            #"ScoreSdeVePipeline",
        ]
    )
    _import_structure["schedulers"].extend(
        [
            #"CMStochasticIterativeScheduler",
            #"DDIMInverseScheduler",
            #"DDIMParallelScheduler",
            "DDIMScheduler",
            #"DDPMParallelScheduler",
            "DDPMScheduler",
            #"DDPMWuerstchenScheduler",
            #"DEISMultistepScheduler",
            #"DPMSolverMultistepInverseScheduler",
            #"DPMSolverMultistepScheduler",
            #"DPMSolverSinglestepScheduler",
            #"EulerAncestralDiscreteScheduler",
            #"EulerDiscreteScheduler",
            #"HeunDiscreteScheduler",
            #"IPNDMScheduler",
            #"KarrasVeScheduler",
            #"KDPM2AncestralDiscreteScheduler",
            #"KDPM2DiscreteScheduler",
            #"LCMScheduler",
            #"PNDMScheduler",
            #"RePaintScheduler",
            "SchedulerMixin",
            #"ScoreSdeVeScheduler",
            #"UnCLIPScheduler",
            #"UniPCMultistepScheduler",
            #"VQDiffusionScheduler",
        ]
    )
    #_import_structure["training_utils"] = ["EMAModel"]


 
if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    from .configuration_utils import ConfigMixin

    try:
        if not is_onnx_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_onnx_objects import *  # noqa F403
    else:
        from .pipelines import OnnxRuntimeModel

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .utils.dummy_pt_objects import *  # noqa F403
    else:
        from .models import (
           
            ModelMixin,
            UNet2DModel,
            #UNet3DConditionModel,
            VQModel,
        )
        from .optimization import (
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
        )
        from .pipelines import (
           
            #DDIMPipeline,
            #DDPMPipeline,
            DiffusionPipeline,
           
            LDMSuperResolutionPipeline,
            
        )
        from .schedulers import (
           
            DDIMScheduler,
            #DDPMParallelScheduler,
            DDPMScheduler,
            SchedulerMixin,
            #VQDiffusionScheduler,
        )
        #from .training_utils import EMAModel
    
else:
#if 0:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
_import_structure["pipelines"].extend("LDMSuperResolutionPipeline")
from .pipelines import LDMSuperResolutionPipeline

