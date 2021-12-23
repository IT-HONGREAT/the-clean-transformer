from cleanformer.tensors import subsequent_mask


def main():

    masked = subsequent_mask(5)
    print(masked)


if __name__ == '__main__':
    main()