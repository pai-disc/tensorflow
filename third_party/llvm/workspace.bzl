"""Provides the repository macro to import LLVM."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo(name):
    """Imports LLVM."""
    LLVM_COMMIT = "aee2a35ac4ab4fe62bb0ce4e314966ab9207efd1"
    LLVM_SHA256 = "348058d7034aac64d7f58ba7711b6fde5d5b3d1bff36f1099fabc9a5c507961e"

    tf_http_archive(
        name = name,
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
            "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        ],
        build_file = "//third_party/llvm:llvm.BUILD",
        patch_file = [
            "//third_party/llvm:build.patch",
            "//third_party/llvm:mathextras.patch",
            "//third_party/llvm:toolchains.patch",
            "//third_party/llvm:temporary.patch",  # Cherry-picks and temporary reverts. Do not remove even if temporary.patch is empty.
            "//third_party/llvm:0001-mlir-ROCm-Add-shfl.sync.bfly-lowering.patch",
            "//third_party/llvm:0001-llvm-nvptx-Fix-error-GVN-on-shared-memory-load.patch",
            "//third_party/llvm:0001-mlir-not-fold-UnrealizedConversionCastOp-with-ui-si.patch",
        ],
        link_files = {"//third_party/llvm:run_lit.sh": "mlir/run_lit.sh"},
    )
