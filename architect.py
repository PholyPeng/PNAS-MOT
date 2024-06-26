import torch
import numpy as np
import torch.nn as nn


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate,
                                          betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)


    def _compute_unrolled_model(self, det_img, det_info, det_id, det_cls, det_split, eta, network_optimizer):
        loss = self.model._loss(det_img, det_info, det_id, det_cls, det_split)
        theta = _concat(self.model.parameters()).detach()
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(
                                 self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).detach() + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, det_info_train, det_id_train, det_cls_train, det_split_train, input_valid, det_info_valid, det_id_valid, det_cls_valid, det_split_valid, eta,
             network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, det_info_train, det_id_train, det_cls_train, det_split_train,
                                         input_valid, det_info_valid, det_id_valid, det_cls_valid, det_split_valid,
                                         eta,
                                         network_optimizer)
        else:
            self._backward_step(input_valid, det_info_valid, det_id_valid, det_cls_valid, det_split_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, det_info_valid, det_id_valid, det_cls_valid, det_split_valid):
        loss = self.model._loss(input_valid, det_info_valid, det_id_valid, det_cls_valid, det_split_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, det_info_train, det_id_train, det_cls_train, det_split_train,
                                        input_valid, det_info_valid, det_id_valid, det_cls_valid, det_split_valid,
                                        eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, det_info_train, det_id_train, det_cls_train, det_split_train, eta,
                                                      network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, det_info_valid, det_id_valid, det_cls_valid, det_split_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.detach() for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, det_info_train, det_id_train, det_cls_train, det_split_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        # does named_parameters contains other parameters in the tracking NetWork that we don't search
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset:offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input_train, det_info_train, det_id_train, det_cls_train, det_split_train, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input_train, det_info_train, det_id_train, det_cls_train, det_split_train)

        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input_train, det_info_train, det_id_train, det_cls_train, det_split_train)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
