import matplotlib.pyplot as plt
import torch

# True parameters for bimodal distribution
MU = 2
SIGMA = 0.5


def log_p(theta):
    """ log of PDF for the true bimodal distribution"""
    log_mode_1 = torch.distributions.Normal(MU, SIGMA).log_prob(theta)
    log_mode_2 = torch.distributions.Normal(-MU, SIGMA).log_prob(theta)

    return torch.logaddexp(log_mode_1, log_mode_2) - torch.log(torch.tensor(2.0))


def sample_p(n):
    """ Samples from the true bimodal distribution """
    modes = torch.randint(0, 2, (n,))
    return torch.where(
        modes == 0,
        MU + SIGMA * torch.randn(n),
        -MU + SIGMA * torch.randn(n)
    )


def log_q(theta, mu, log_sigma):
    """ log of PDF for the predicted distribution """
    sigma = torch.exp(log_sigma)
    return torch.distributions.Normal(mu, sigma).log_prob(theta)


def sample_q(mu, log_sigma, n):
    """ Samples from the predicted distribution """
    sigma = torch.exp(log_sigma)
    return torch.distributions.Normal(mu, sigma).rsample((n,))


def forward_kl_loss(mu, log_sigma, n):
    """ Estimates the forward KL divergence """
    theta = sample_p(n)

    log_p_theta = log_p(theta)
    log_q_theta = log_q(theta, mu, log_sigma)

    return torch.mean(log_p_theta - log_q_theta)


def reverse_kl_loss(mu, log_sigma, n):
    """ Estimates the reverse KL divergence """
    theta = sample_q(mu, log_sigma, n)

    log_p_theta = log_p(theta)
    log_q_theta = log_q(theta, mu, log_sigma)

    return torch.mean(log_q_theta - log_p_theta)


def vector_field(theta, t, mu, sigma, sigma_min=0.01):
    """ Returns the predicted vector field"""
    sigma_t = 1 - (1 - sigma_min) * t

    numerator = (sigma_t ** 2 + (t * sigma) ** 2 - sigma_t) * theta + t * mu * sigma_t
    denominator = t * (sigma_t ** 2 + (t * sigma) ** 2) + 1e-8

    return numerator / denominator


def flow_matching_loss(mu, log_sigma, n, sigma_min=0.01):
    """ Estimates the flow matching loss"""
    sigma = torch.exp(log_sigma)

    t = torch.rand(n)
    theta_1 = sample_p(n)

    mu_t = theta_1 * t
    sigma_t = 1 - (1 - sigma_min) * t
    theta_t = mu_t + sigma_t * torch.randn(n)

    v_t = vector_field(theta_t, t, mu, sigma, sigma_min=sigma_min)
    u_t = (theta_1 - (1 - sigma_min) * theta_t) / (1 - (1 - sigma_min) * t)

    loss = torch.nn.functional.mse_loss(v_t, u_t)

    return loss


def train():
    EPOCHS = 1000
    LR = 1e-2
    SAMPLES = 100_000  # Number of samples per epoch

    losses = []

    mu = torch.tensor(-2., requires_grad=True)
    log_sigma = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.Adam([mu, log_sigma], lr=LR)

    for step in range(EPOCHS):
        optimizer.zero_grad()

        loss = flow_matching_loss(mu, log_sigma, n=SAMPLES)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Epoch {step}: {loss.item():.4f}, MU: {mu.item():.4f}, log_sigma: {log_sigma.item():.4f}")

    print("MU:", mu.detach().numpy())
    print("SIGMA:", log_sigma.exp().detach().numpy())


def main():
    # train()

    import numpy as np
    from scipy.stats import norm

    # Solving the ODE
    n_steps = 100
    n_samples = 25

    pct = np.linspace(0.001, 0.999, n_samples)
    x = norm.ppf(pct)

    paths = np.zeros((n_steps + 1, n_samples))
    paths[0] = x.copy()

    T = 0.0
    dt = 1 / n_steps

    for step in range(1, n_steps + 1):
        T += dt
        v_t = vector_field(x, T, 0.0, 2.06)

        x += v_t * dt
        paths[step] = x.copy()

    # Plotting the particle paths
    plt.figure(figsize=(12, 8))
    for sample in range(n_samples):
        plt.plot(
            np.linspace(0, 1, n_steps + 1),
            paths[:, sample], color="tab:blue", linewidth=1, label="Vector Field" if sample == 0 else None
        )

    y = np.linspace(-5, 5, 100)
    plt.plot((norm(-2, 0.5).pdf(y) + norm(2, 0.5).pdf(y))*0.2, y, color="tab:gray", linestyle="--", label="True Posterior")
    plt.plot(1.0 - (norm(-2, 0.5).pdf(y) + norm(2, 0.5).pdf(y))*0.2, y, color="tab:gray", linestyle="--")
    plt.plot(norm.pdf(y) * 0.4, y, color="black", label="Predicted Posterior")
    plt.plot(1.0 - norm(0, 2.06).pdf(y) * 0.4, y, color="black")
    plt.ylim(-4, 4)

    plt.xlabel(r"Time $t$")
    plt.ylabel(r"$\theta$")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
