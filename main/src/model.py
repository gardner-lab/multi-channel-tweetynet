import torch


class TweetynetModel:
    def __init__(
        self, device,
    ):
        self.device = device
        self.global_step = 0
        self.loss_func = torch.nn.CrossEntropyLoss().cuda(device)

    def run(self, train_data, eval_data, net, num_epochs, eval_step, lr):
        optimizer = torch.optim.Adam(params=net.parameters(), lr=lr)
        accs, xs = self.train(
            eval_step=eval_step,
            num_epochs=num_epochs,
            train_data=train_data,
            eval_data=eval_data,
            net=net,
            optim=optimizer,
        )
        del optimizer
        return xs, accs

    def train(self, eval_step, num_epochs, train_data, eval_data, net, optim):
        self.global_step = 0

        net.train()

        accs = []
        num_samps = []

        for epoch in range(1, num_epochs + 1):
            print(f"---------------- EPOCH {epoch} \n\n")

            for batch in train_data:
                x, y = (
                    batch[0],
                    batch[1].cuda(self.device, non_blocking=True),
                )
                for nfft, spec in x.items():
                    size = spec.size()
                    x[nfft] = spec.view(size[0], 1, size[1], size[2]).cuda(
                        self.device, non_blocking=True
                    )

                y_pred = net.forward(x).to(self.device)
                optim.zero_grad()
                loss = self.loss_func(y_pred, y)
                loss.backward()
                optim.step()

                self.global_step += train_data.batch_size
                num_samps.append(self.global_step)

                if self.global_step % eval_step == 0:
                    eval_acc = self.eval(net, eval_data)
                    accs.append(eval_acc)
                    net.train()

            update = f"""
            Number Training Samples: {self.global_step}
            Eval Accuracy: {eval_acc}
            \n
            """
            print(update)

        return accs, num_samps

    def eval(self, net, eval_data):
        net.eval()

        accuracies = []

        with torch.no_grad():
            for batch in eval_data:
                x, y = (
                    batch[0],
                    batch[1].to(self.device),
                )
                for nfft, spec in x.items():
                    size = spec.size()
                    x[nfft] = spec.view(size[1], size[0], size[2], size[3]).to(
                        self.device
                    )

                y_pred = net.forward(x).to(self.device)
                accuracies.append(self.accuracy(y_pred, y))

        accuracy = sum(accuracies) / len(accuracies)
        return accuracy

    def accuracy(self, y_pred, y):
        y_pred = y_pred.permute(1, 0, 2)
        y_pred = torch.flatten(y_pred, start_dim=1)
        y_pred = torch.unsqueeze(y_pred, 0)
        y_pred = torch.argmax(y_pred, dim=1)
        if y.size()[0] == 1:
            y = torch.squeeze(y)
        y = torch.flatten(y, 0)
        y = torch.unsqueeze(y, 0)
        correct = torch.eq(y_pred, y).view(-1)
        return correct.sum().item() / correct.size()[0]
