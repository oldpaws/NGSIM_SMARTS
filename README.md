# How to run

Step 0. Download the SMARTS submodule with `git submodule init & git submodule update`, **make sure your SMARTS is at the right commit id (46c3497)**

Step 1. Install SMARTS following the instructions in `./SMARTS` folder

Step 2. Download the original data from <https://jbox.sjtu.edu.cn/v/link/view/75a0931a222347e1ba2e0441407f4a1f> and place it under `./ngsim` folder

Step 3. Build NGSIM scenario with `scl scenario build --clean ./ngsim` (This will take a while)

Step 4. Generate expert demonstrations with `python example_expert_generation.py`

Step 5. Test rollout with `python example_rollout.py` (Alternatively, to accelerate sampling, you can check `parallel_rollout.py`).

# Setup using Docker

If you have trouble with Step 1, you can configure SMARTS with docker. 

1. Build the image

```bash
docker build . -t ngsim
```

2. Create the container
```bash
docker run -it -v $PWD:/src -p 8081:8081 --name <your_container_name> ngsim bash
```

Then you can continue from Step 2 inside the docker container.


# Benchmark your policy

We provide a benchmark script for you to evaluate your model locally. Now the example script is just performing random actions policy, you can customize for your own policy models.

You can run the script with the following command:

```bash
python benchmark/benchmark.py <path_to_your_ngsim_scenario> --env_num 10
```

<!-- 
# Troubleshooting

1. **TypeError: export_glb() got an unexpected keyword argument 'extras'**
  
    Try install an alternative version of trimesh with:
    ```bash
    pip install trimesh==3.9.20
    ``` -->
