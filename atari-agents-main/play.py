from argparse import ArgumentParser
from functools import partial
from gzip import GzipFile
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from numpy.testing.print_coercion_tables import print_new_cast_table
from torch import nn
import torchvision
import torchvision.transforms.v2 as trans
from torchvision.transforms import InterpolationMode
import pickle

from ale_env import ALEModern, ALEClassic


class AtariNet(nn.Module):
    """ Estimator used by DQN-style algorithms for ATARI games.
        Works with DQN, M-DQN and C51.
    """
    def __init__(self, action_no, distributional=False):
        super().__init__()

        self.action_no = out_size = action_no
        self.distributional = distributional

        # configure the support if distributional
        if distributional:
            support = torch.linspace(-10, 10, 51)
            self.__support = nn.Parameter(support, requires_grad=False)
            out_size = action_no * len(self.__support)

        # get the feature extractor and fully connected layers
        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, out_size),
        )

    def forward(self, x):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)

        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))

        if self.distributional:
            logits = qs.view(qs.shape[0], self.action_no, len(self.__support))
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.__support.expand_as(qs_probs)).sum(2)
        return qs


def _load_checkpoint(fpath, device="cpu"):
    print(fpath)
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            print(type(inflated))
            return torch.load(inflated, map_location=device)
    # with fpath.open('rb') as file:
    #     return torch.load(file, weights_only=False)


def _epsilon_greedy(obs, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    q_val, argmax_a = model(obs).max(1)
    return argmax_a.item(), q_val

def motion_blur(prev_obs, obs):
    # Motion blur, higher alpha = more blur
    prev_obs = prev_obs.numpy()
    obs = obs.numpy()
    alpha = 0.8
    beta = (1.0 - alpha)
    blurred = cv2.addWeighted(prev_obs, alpha, obs, beta, 0.0)
    blurred = torch.tensor(blurred)
    return blurred

def gaussian_blur(obs):
    return trans.GaussianBlur(5, sigma=0.3)(obs)

def gaussian_noise(obs):
    return trans.GaussianNoise(sigma=0.05)(obs)

def random_erase(obs):
    return trans.RandomErasing(p=0.6, scale=(0.3, 0.6), ratio=(0.5, 2.0))(obs)

def earthquake(obs, angle):
    return trans.RandomAffine(degrees=(-angle, angle))(obs)

# Assuming an 84 X 84 original picture
def pixelate(obs, target_res):
    obs = trans.Resize(target_res, interpolation=InterpolationMode.NEAREST_EXACT)(obs)
    return trans.Resize((84, 84), interpolation=InterpolationMode.NEAREST_EXACT)(obs)

def generate_random_lines(imshape, slant, drop_length):
    drops = []
    for i in range(10): ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant,imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x,y))
    return drops

def add_rain(observation, slant):
    drop_length = 3
    drop_width = 1
    rain_drops = generate_random_lines(observation[0][0].numpy().shape, slant, drop_length)
    for i in range(len(observation[0])):
        image = observation[0][i].numpy()
        imshape = image.shape
        for rain_drop in rain_drops:
            cv2.line(image, (rain_drop[0], rain_drop[1]),
                     (rain_drop[0] + slant, rain_drop[1] + drop_length), 200, drop_width)
        observation[0][i] = torch.tensor(image)
    return observation

def main(opt):
    # game/seed/model
    ckpt_path = Path(opt.path)
    game = ckpt_path.parts[-3]

    # recording
    if opt.record:
        record_dir = Path.cwd() / "movies" / Path(*ckpt_path.parts[-4:-1])
        record_dir.mkdir(parents=True, exist_ok=False)
        print("Recording@ ", record_dir)

    # set env
    ALE = ALEModern if "_modern/" in opt.path else ALEClassic
    env = ALE(
        game,
        torch.randint(100_000, (1,)).item(),
        sdl=True,
        device="cpu",
        clip_rewards_val=False,
        record_dir=str(record_dir) if opt.record else None,
    )

    if opt.variations:
        env.set_mode_interactive()

    # init model
    model = AtariNet(env.action_space.n, distributional="C51_" in opt.path)

    # sanity check
    print(env)

    # load state
    ckpt = _load_checkpoint(opt.path)
    model.load_state_dict(ckpt["estimator_state"])

    # configure policy
    policy = partial(_epsilon_greedy, model=model, eps=0.001)
    ep_returns = [0 for _ in range(opt.episodes)]

    # Generating an angle for rain to fall at
    slant_extreme = 2
    slant = np.random.randint(-slant_extreme, slant_extreme)
    # slant = -2

    for ep in range(opt.episodes):
        obs, done = env.reset(), False
        frames = []
        actions = []
        rewards = []
        while not done:
            action, _ = policy(obs)
            frames.append(obs[0][0])
            actions.append(action)
            rewards.append(ep_returns[ep])

            # Uncomment line below for non-frozen normal functioning
            prev_obs = obs

            obs, reward, done, _ = env.step(action)

            # Code for frozen frames    # TODO: extract to method
            # if random.randint(0, 2) != 0:
            #     obs = prev_obs
            # else:
            #     prev_obs = obs

            # obs = motion_blur(prev_obs, obs)
            # obs = gaussian_blur(obs)
            # obs = gaussian_noise(obs)
            # obs = random_erase(obs)
            # obs = pixelate(obs, (42, 42))
            # obs = earthquake(obs, 50)
            # obs = add_rain(obs, slant)

            # TODO: fix upscaling so it shows exactly what the agent sees (no upscaling or blurring)
            cv2.imshow("yo", cv2.resize(obs.numpy()[0][0], [420, 420], interpolation=cv2.INTER_LINEAR))
            ep_returns[ep] += reward
        print(f"{ep:02d})  Gt: {ep_returns[ep]:7.1f}")

        frames_tensor = torch.stack(frames, dim=0)
        actions_tensor = torch.tensor(actions)
        rewards_tensor = torch.tensor(rewards)
        torch.save(frames_tensor, f'recordings/episode{ep}_frames.pt')
        torch.save(actions_tensor, f'recordings/episode{ep}_actions.pt')
        torch.save(rewards_tensor, f'recordings/episode{ep}_rewards.pt')





if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("game", type=str, help="game name")
    parser.add_argument("path", type=str, help="path to the model")
    parser.add_argument(
        "-e", "--episodes", default=10, type=int, help="number of episodes"
    )
    parser.add_argument(
        "-v",
        "--variations",
        action="store_true",
        help="set mode and difficulty, interactively",
    )
    parser.add_argument(
        "-r", "--record", action="store_true", help="record png screens and sound",
    )

    main(parser.parse_args())


