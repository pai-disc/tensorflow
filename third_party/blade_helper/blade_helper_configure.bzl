load("//third_party/remote_config:common.bzl", "get_host_environ")

_TAO_BUILD_VERSION = "TAO_BUILD_VERSION"
_TAO_BUILD_GIT_BRANCH = "TAO_BUILD_GIT_BRANCH"
_TAO_BUILD_GIT_HEAD = "TAO_BUILD_GIT_HEAD"
_TAO_BUILD_HOST = "TAO_BUILD_HOST"
_TAO_BUILD_IP = "TAO_BUILD_IP"
_TAO_BUILD_TIME = "TAO_BUILD_TIME"


def _blade_helper_impl(repository_ctx):
    repository_ctx.template("build_defs.bzl", Label("//bazel/blade_helper:build_defs.bzl.tpl"), {
        "%{TAO_BUILD_VERSION}": get_host_environ(repository_ctx, _TAO_BUILD_VERSION),
        "%{TAO_BUILD_GIT_BRANCH}": get_host_environ(repository_ctx, _TAO_BUILD_GIT_BRANCH),
        "%{TAO_BUILD_GIT_HEAD}": get_host_environ(repository_ctx, _TAO_BUILD_GIT_HEAD),
        "%{TAO_BUILD_HOST}": get_host_environ(repository_ctx, _TAO_BUILD_HOST),
        "%{TAO_BUILD_IP}": get_host_environ(repository_ctx, _TAO_BUILD_IP),
        "%{TAO_BUILD_TIME}": get_host_environ(repository_ctx, _TAO_BUILD_TIME),
    })

    repository_ctx.template("BUILD", Label("//bazel/blade_helper:BUILD.tpl"), {
    })

blade_helper_configure = repository_rule(
    implementation = _blade_helper_impl,
    environ = [
        _TAO_BUILD_VERSION,
        _TAO_BUILD_GIT_BRANCH,
        _TAO_BUILD_GIT_HEAD,
        _TAO_BUILD_HOST,
        _TAO_BUILD_IP,
        _TAO_BUILD_TIME
    ],
)
