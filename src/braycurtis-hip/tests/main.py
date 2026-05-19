import sklearn.metrics.pairwise as sklearn_pw
import torch
import torchpairwise

def test_unbatch (dtype=torch.float32, device='cuda'):
    print(f'Compute unbatched braycurtis distance with data type {dtype}')
    torch.manual_seed(0)
    x1 = torch.rand(2000, 768, dtype=dtype);
    x2 = torch.rand(5000, 768, dtype=dtype);
    sk_output = sklearn_pw.pairwise_distances(x1, x2, metric='braycurtis')
    x1 = x1.to(device)
    x2 = x2.to(device)
    pw_output = torchpairwise.braycurtis_distances(x1, x2).detach().cpu()
    print(f'The maximum difference is '
          f'{torch.max(torch.abs(pw_output - sk_output))}')

def test_batch (dtype=torch.float32, device='cuda'):
    print(f'Compute batched braycurtis distance with data type {dtype}')
    torch.manual_seed(0)
    b = 8  # batch
    x1 = torch.rand(b, 1000, 768, dtype=dtype);
    x2 = torch.rand(b, 2500, 768, dtype=dtype);
    sk_output = []
    for i in range(b):
      # returns ndarray
      t = sklearn_pw.pairwise_distances(x1[i], x2[i], metric='braycurtis')
      sk_output.append(torch.tensor(t))
    sk_output = torch.stack(sk_output, 0); # stack a list of 2d torch tensors
    x1 = x1.to(device)
    x2 = x2.to(device)
    pw_output = torchpairwise.braycurtis_distances(x1, x2).detach().cpu()
    print(f'The maximum difference is '
          f'{torch.max(torch.abs(pw_output - sk_output))}')

def check_grad (dtype=torch.float64, device='cuda'):
    torch.manual_seed(0)
    x1 = torch.rand(2, 2, dtype=dtype, device=device);
    x2 = torch.rand(2, 2, dtype=dtype, device=device);
    x1.requires_grad_()
    x2.requires_grad_()
    grad_correct = torch.autograd.gradcheck(
        lambda x, y: torchpairwise.braycurtis_distances(x, y), inputs=(x1, x2), raise_exception=False)
    print('grad_correct:', grad_correct)

    x1 = torch.rand(2, 1, 2, dtype=dtype, device=device);
    x2 = torch.rand(2, 1, 2, dtype=dtype, device=device);
    x1.requires_grad_()
    x2.requires_grad_()
    grad_correct = torch.autograd.gradcheck(
        lambda x, y: torchpairwise.braycurtis_distances(x, y), inputs=(x1, x2), raise_exception=False)
    print('grad_correct:', grad_correct)


if __name__ == '__main__':
    check_grad(dtype=torch.float64)  # ooo issue for large inputs
    test_unbatch(dtype=torch.float32)
    test_unbatch(dtype=torch.float64)
    test_batch(dtype=torch.float32)
    test_batch(dtype=torch.float64)
