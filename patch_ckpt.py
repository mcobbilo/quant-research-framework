import torch

def patch_checkpoint(ckpt_path):
    print(f"Loading checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    patched = False

    if "hyper_parameters" in ckpt:
        hparams = ckpt["hyper_parameters"]
        if "monotone_constaints" in hparams:
            print("Found 'monotone_constaints' typo inside checkpoint. Fixing to 'monotone_constraints'...")
            val = hparams.pop("monotone_constaints")
            # Usually it expects proper constraints dict, we can set it to default or preserve it if needed
            hparams["monotone_constraints"] = val
            patched = True
            
    if patched:
        torch.save(ckpt, ckpt_path)
        print("Successfully saved patched checkpoint over original file.")
    else:
        print("No typographic anomalies found inside binary.")

if __name__ == "__main__":
    patch_checkpoint("/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework/lightning_logs/version_1166/checkpoints/epoch=10-step=2596.ckpt")
