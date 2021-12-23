import torch


def main():
    L = 10
    ones = torch.ones(size=(L,L))
    print(ones)
    tirled = torch.tril(ones,diagonal=0)
    print("tirled : \n",tirled)



if __name__ == '__main__':
    main()