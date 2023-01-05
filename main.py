import cv2
import matplotlib.pyplot as plt
import numpy as np

from py_wake.site.xrsite import UniformSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake import NOJ


def read():
    cap = cv2.VideoCapture(0)
    cap.set(3,240) #width=640
    cap.set(4,240) #height=480
    _,frame = cap.read()
    cap.release()
    return frame


def load_image():
    img = cv2.imread('img.png')
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 


def find_positions(img, threshold=70, step=10):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 5)
    _, thresh = cv2.threshold(img, threshold, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return np.array([(np.mean(c[:, 0, 0]).astype(int), np.mean(c[:, 0, 1]).astype(int)) for c in contours]) * step


def get_wake_loss(positions, turbine=V80(), site=UniformSite()):
    calculator = NOJ(site, turbine)
    simulationResult = calculator(positions[:, 0], positions[:, 1])
    return 1 - simulationResult.aep().sum() / simulationResult.aep(with_wake_loss=False).sum()


if __name__ == "__main__":
    img = load_image()

    positions = find_positions(img)

    print (f"Number of turbines found: {len(positions)}")

    loss = get_wake_loss(positions)

    print (f"Total wake loss: {loss * 100:.1f} %")

    plt.imshow(img)
    for p in positions:
        plt.scatter(p[0], p[1])
    plt.show()
