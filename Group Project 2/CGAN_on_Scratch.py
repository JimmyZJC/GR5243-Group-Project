import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skopt import gp_minimize
from skopt.space import Real, Integer

class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        shortcut = self.shortcut(x)
        return nn.ReLU()(out + shortcut)

class NumericalDGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim, hidden_dim):
        super(NumericalDGenerator, self).__init__()
        self.layers = nn.Sequential(
            ResidualLayer(noise_dim + label_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim = 1)
        return self.layers(x)

class CategoricalGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, output_dim, hidden_dim):
        super(CategoricalGenerator, self).__init__()
        self.layers = nn.Sequential(
            ResidualLayer(noise_dim + label_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim = 1)
        return self.layers(x)

class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, num_output_numerical, num_output_categorical, hidden_dim):
        super(Generator, self).__init__()
        self.numerical_gen = NumericalDGenerator(noise_dim, label_dim, num_output_numerical, hidden_dim)
        self.categorical_gen = CategoricalGenerator(noise_dim, label_dim, num_output_categorical, hidden_dim)

    def forward(self, noise, labels):
        numerical_data = self.numerical_gen(noise, labels)
        categorical_data = self.categorical_gen(noise, labels)
        return torch.cat([numerical_data, categorical_data], dim = 1)

class NumericalDiscriminator(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim):
        super(NumericalDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            ResidualLayer(input_dim + label_dim, hidden_dim),
        )
    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim = 1)
        return self.layers(x)

class CategoricalDiscriminator(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim):
        super(CategoricalDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            ResidualLayer(input_dim + label_dim, hidden_dim),
        )
    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim = 1)
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, label_dim, num_input_numerical, num_input_categorical, hidden_dim):
        super(Discriminator, self).__init__()
        self.num_input_numerical = num_input_numerical
        self.numerical_disc = NumericalDiscriminator(num_input_numerical, label_dim, hidden_dim)
        self.categorical_disc = CategoricalDiscriminator(num_input_categorical, label_dim, hidden_dim)
        self.final_layer = nn.Sequential(
            ResidualLayer(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            ResidualLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data, labels):
        numerical_data, categorical_data = data[:, :self.num_input_numerical], data[:, self.num_input_numerical:]
        numerical_fetaure = self.numerical_disc(numerical_data, labels)
        categorical_feature = self.categorical_disc(categorical_data, labels)
        features = torch.cat([numerical_fetaure, categorical_feature], dim = 1)
        return self.final_layer(features)

def he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode = "fan_in", nonlinearity = "relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def train_cgan(generator, discriminator, train_loader, device, gen_opt, disc_opt, num_epochs, noise_dim):
    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        for i, (real_data, labels) in enumerate(train_loader):
            real_data, labels = real_data.to(device), labels.to(device)
            batch_size = real_data.size(0)

            disc_opt.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device = device)
            fake_labels = torch.zeros(batch_size, 3, device=device)
            random_indices = torch.randint(0, 3, (batch_size,), device=device)
            fake_labels.scatter_(1, random_indices.unsqueeze(1), 1)
            fake_data = generator(noise, fake_labels.to(device))
            real_output = discriminator(real_data, labels.to(device))
            fake_output = discriminator(fake_data.detach(), fake_labels.to(device))
            d_loss = criterion(real_output, torch.ones_like(real_output)) + criterion(fake_output, torch.zeros_like(fake_output))
            d_loss.backward()
            disc_opt.step()

            gen_opt.zero_grad()
            fake_output = discriminator(fake_data, fake_labels.to(device))
            g_loss = criterion(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            gen_opt.step()
    return g_loss.item()

def bayesian_optimization_cgan(train_loader, device, noise_dim, label_dim, num_numerical, num_categorical):
    def objective(params):
        num_epochs, gen_hidden_dim, disc_hidden_dim, gen_lr, disc_lr, gen_beta1, disc_beta1 = params

        generator = Generator(noise_dim, label_dim, num_numerical, num_categorical, gen_hidden_dim).to(device)
        discriminator = Discriminator(label_dim, num_numerical, num_categorical, disc_hidden_dim).to(device)
        generator.apply(he_init)
        discriminator.apply(he_init)
        gen_optimizer = optim.Adam(generator.parameters(), gen_lr, betas = (gen_beta1, 0.999))
        disc_optimizer = optim.Adam(discriminator.parameters(), lr = disc_lr, betas = (disc_beta1, 0.999))

        gloss = train_cgan(generator, discriminator, train_loader, device, gen_optimizer, disc_optimizer, num_epochs, noise_dim)
        return -gloss

    search_space = [
        Integer(50, 100),
        Integer(128, 512),
        Integer(128, 512),
        Real(1e-5, 1e-1, prior="log-uniform"),
        Real(1e-5, 1e-1, prior="log-uniform"),
        Real(0.0, 0.999),
        Real(0.0, 0.999)
    ]

    result = gp_minimize(
        func = objective,
        dimensions = search_space,
        n_calls = 100,
        random_state = 233,
        n_jobs = -1,
        verbose = 0
    )
    
    best_params = {
        "num_epochs": result.x[0],
        "gen_hidden_dim": result.x[1],
        "disc_hidden_dim": result.x[2],
        "gen_lr": result.x[3],
        "disc_lr": result.x[4],
        "gen_beta1": result.x[5],
        "disc_beta1": result.x[6],
    }

    return best_params