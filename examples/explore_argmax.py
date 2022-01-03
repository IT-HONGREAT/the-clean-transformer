import torch


def main():

    x = torch.Tensor([[1,10,5,9,100],[90,90,980,80,80]])
    print(torch.argmax(x,dim=-1))

if __name__ == '__main__':
    main()