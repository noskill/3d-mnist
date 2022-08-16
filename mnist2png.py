import numpy
from torchvision import datasets, transforms


def main():
    dataset1 = datasets.MNIST('./data', train=True, download=True)

    dataset2 = datasets.MNIST('./data', train=False)
    save_dataset(dataset1, 'train')
    save_dataset(dataset2, 'test')

def save_dataset(dataset1, test_train):
    for i, (pic, cls) in enumerate(dataset1):
        num = '0' * (5 - len(str(i))) + str(i)
        arr = numpy.array(pic)
        pic.save("./{0}/img_{1}_{2}.png".format(test_train, num, cls))



main()

