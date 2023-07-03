# Model Examples

## TOC
<!-- TOC -->

- [Model Examples](#model-examples)
    - [TOC](#toc)
    - [Example list](#example-list)
    - [Run an example](#run-an-example)
    - [Create new examples](#create-new-examples)
        - [Troubleshoots](#troubleshoots)

<!-- /TOC -->

## Example list
- Deep Neural Network:
    - Typical layered neural network.
    - Folder: [./DNN](./DNN)
    - Entry file: [./DNN/train.jl](./DNN/train.jl)
    - Model definition: [./DNN/model.jl](./DNN/model.jl)
- Recursive Tree:
    - Dealing with tree-structure data, using recursive computation.
    - Folder: [./Tree](./Tree)
    - Entry file: [./Tree/train.jl](./Tree/train.jl)
    - Model definition: [./Tree/model.jl](./Tree/model.jl)

## Run an example
Each example is wrapped in a standalone Julia package, which has the main AD tool and other shared
utilities as depended packages.

To run an example, we should:
- Change the working directory to its folder;
- Restore depended packages;
    - This should be done for each example when running it for the first time.
- Run its entry file, with its package active.

For example, to run the sample model under `./DNN`, in the shell we could run:

```shell
cd ./DNN

# Resolve depended packages
julia --project -e "using Pkg; Pkg.instantiate()"

# Run the entry file
julia --project train.jl
```

## Create new examples
New examples should:
1. have a clear package description and metadata;
1. have references to main AD package and shared package `Utils`.

To create a new example:
1. Under `/examples` folder, enter Julia `[`-mode REPL (package management REPL, by pressing
`[` key in normal Julia REPL), run:
    ```shell
    generate SAMPLE_NAME
    ```

    This will create a new folder `/examples/SAMPLE_NAME`.

1. Exit Julia REPL, change the working directory in the shell to the newly created
`/examples/SAMPLE_NAME`.

1. Start Julia REPL again, with the package in the current working directory being the active
package, by running:
    ```shell
    julia --project
    ```

1. Enter `[`-mode REPL and add references to main AD package and `Utils` package:
    ```julia
    dev ../..
    dev ../Utils
    ```

    Note: `dev` command in `[`-mode REPL is to add a package for development purpose. This does not require that package to be published or its changes to be committed into Git history.

### Troubleshoots
In `[`-mode REPL with some example package activated, run `resolve` and `instantiate` if there is an issue related to package environment.
