import os
import shutil
import yaml
from unet.train import Trainer

DEVICE = os.getenv("DEVICE", None)
assert DEVICE is not None, "Please specife env var `DEVICE`"


def main():
    with open("config_template.yml") as f:
        config = yaml.load(f)
    exp_path = os.path.join("experiments", config["experiment"])
    if os.path.exists(exp_path):
        ans = None
        while ans != "y" and ans != "n":
            ans = input("Path [{}] exists do you want to delete?[Y/n]".format(exp_path))
        if ans == "n":
            return
        shutil.rmtree(exp_path)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    with open(os.path.join(exp_path, "config.yml"), "w") as f:
        yaml.dump(config, f)
    trainer = Trainer(exp_path, config, device=DEVICE)
    trainer.train()


if __name__ == '__main__':
    main()
