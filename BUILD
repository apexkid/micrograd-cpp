cc_library(
    name = "micrograd",
    srcs = ["micrograd.cc"],
    hdrs = ["micrograd.h"],
    deps = [
        # Add dependencies here if any
    ],
)

cc_binary(
    name = "micrograd_main",
    srcs = ["micrograd_main.cc"],
    deps = [":micrograd"], 
)

cc_test(
    name = "micrograd_test",
    srcs = ["micrograd_test.cc"],
    deps = [
        ":micrograd",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "nn_linear_regression_demo",
    srcs = ["nn_linear_regression_demo.cc"],
    deps = [":micrograd"], 
)

cc_binary(
    name = "nn_logistic_regression_demo",
    srcs = ["nn_logistic_regression_demo.cc"],
    deps = [":micrograd"], 
)