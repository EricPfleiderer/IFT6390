
import matplotlib.pyplot as plt


def show_seq(x):
    plt.figure()
    plt.plot(range(len(x)), x)
    plt.show()


def show_seqs(x):
    plt.figure()

    for seq in x:
        plt.plot(range(len(seq)), seq)

    plt.show()
