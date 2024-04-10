# @title investigator.py

import torch
import numpy as np

from app.utils.lrp_inverter_util import RelevancePropagator
from app.utils.lrp_utils import pprint, Flatten

class InnvestigateModel(torch.nn.Module):
    """
    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If., for example,
        only the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self,
                 the_model,
                 lrp_exponent=1,
                 beta=.5,
                 epsilon=1e-6,
                 method="e-rule"):

        super(InnvestigateModel, self).__init__()
        self.model = the_model
        #self.device = torch.device("cpu", 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #vag
        self.model.to(self.device) #vag
        self.prediction = None
        self.r_values_per_layer = None
        self.only_max_score = None
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            beta=beta, method=method, epsilon=epsilon,
                                            device=self.device)

        # Parsing the individual model layers
        self.register_hooks(self.model)
        if method == "b-rule" and float(beta) in (-1., 0):
            which = "positive" if beta == -1 else "negative"
            which_opp = "negative" if beta == -1 else "positive"
            print("WARNING: With the chosen beta value, "
                  "only " + which + " contributions "
                  "will be taken into account.\nHence, "
                  "if in any layer only " + which_opp +
                  " contributions exist, the "
                  "overall relevance will not be conserved.\n")

    def cuda(self, device=None):
        self.device = torch.device("cuda", device)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cuda(device)

    def cpu(self):
        self.device = torch.device("cpu", 0)
        self.inverter.device = self.device
        return super(InnvestigateModel, self).cpu()

    def register_hooks(self, parent_module):
        """
        Recursively unrolls a model and registers the required
        hooks to save all the necessary values for LRP in the forward pass.

        Args:
            parent_module: Model to unroll and register hooks for.

        Returns:
            None

        """
        for mod in parent_module.children():
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(
                self.inverter.get_layer_fwd_hook(mod))
            if isinstance(mod, torch.nn.ReLU):
                mod.register_backward_hook(
                    self.relu_hook_function
                )

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):
        """
        The innvestigate wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.

        Args:
            in_tensor: Model input to pass through the pytorch model.

        Returns:
            Model output.
        """
        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor):
        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.

        Args:
            in_tensor: New input for which to predict an output.

        Returns:
            Model prediction
        """
        # Reset module list. In case the structure changes dynamically,
        # the module list is tracked for every forward pass.
        self.inverter.reset_module_list()
        in_tensor.to(self.device)
        _ , self.prediction = self.model(in_tensor)
        return self.prediction

    def get_r_values_per_layer(self):
        if self.r_values_per_layer is None:
            pprint("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
        return self.r_values_per_layer

    def innvestigate(self, in_tensor=None, rel_for_class=None):
        """
        DELETED BY ME
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                org_shape = self.prediction.size()
                #print('org_shape', org_shape)
                # Make sure shape is just a 1D vector per batch example.
                #self.prediction = self.prediction.view(org_shape[0], -1)
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction', self.prediction)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                #print('only_max_score', only_max_score)
                relevance_tensor = only_max_score.view(org_shape)
                #print('relevance_tensor', relevance_tensor)
                self.prediction.view(org_shape)

            else:
                org_shape = self.prediction.size()
                self.prediction = self.prediction.view(org_shape[0], -1)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            #print('All layers (rev_model)', rev_model)
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:#[:2]:
                # Compute layer specific backwards-propagation of relevance values
                #print('layer in the for loop', layer)
                #print('relevance shape', relevance.shape)
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                #print('relevance shape', relevance.shape)
                #print(layer, ' computed succesfully')
                r_values_per_layer.append(relevance) #relevance.cpu()

            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return self.prediction, r_values_per_layer[-1]

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()
