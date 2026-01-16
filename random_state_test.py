import torch
import torchvision.transforms.v2 as transforms
import cv2
import numpy as np
import random

# Pick random seed and fix that
# seed = np.random.randint(2147483647)
random.seed(10)
torch.manual_seed(10)

while True:
    test = np.ones((256,256))
    test = test * 256
    test = torch.from_numpy(test)
    test = transforms.RandomErasing(p=1.0, scale=(0.3, 0.6), ratio=(0.5, 2.0))(test)
    test = test.numpy()

    second = np.ones((256,256))
    second = second * 256
    second = torch.from_numpy(second)
    second = transforms.RandomErasing(p=1.0, scale=(0.3, 0.6), ratio=(0.5, 2.0))(second)
    second = second.numpy()

    cv2.imshow("one", test)
    cv2.imshow("second", second)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
