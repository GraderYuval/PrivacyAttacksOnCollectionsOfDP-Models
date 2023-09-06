from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
import numpy as np
from typing import Optional
from art.utils import check_and_transform_label_format


class Attack(LabelOnlyDecisionBoundary):
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs):
        """
                Infer membership of input `x` in estimator's training data.

                :param x: Input data.
                :param y: True labels for `x`.
                :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                                      the predicted class

                :Keyword Arguments for HopSkipJump:
                    * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.
                    * *max_iter*: Maximum number of iterations.
                    * *max_eval*: Maximum number of evaluations for estimating gradient.
                    * *init_eval*: Initial number of evaluations for estimating gradient.
                    * *init_size*: Maximum number of trials for initial generation of adversarial examples.
                    * *verbose*: Show progress bars.

                :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                         or class probabilities.
                """
        from art.attacks.evasion.hop_skip_jump import HopSkipJump

        if y is None:  # pragma: no cover
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")

        if self.distance_threshold_tau is None:  # pragma: no cover
            raise ValueError(
                "No value for distance threshold `distance_threshold_tau` provided. Please set"
                "`distance_threshold_tau` or run method `calibrate_distance_threshold` on known training and test"
                "dataset."
            )

        if "probabilities" in kwargs:
            probabilities = kwargs.get("probabilities")
            del kwargs["probabilities"]
        else:
            probabilities = False

        if "classifier" in kwargs:  # pragma: no cover
            raise ValueError("Keyword `classifier` in kwargs is not supported.")

        if "targeted" in kwargs:  # pragma: no cover
            raise ValueError("Keyword `targeted` in kwargs is not supported.")

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)
        x_adv = hsj.generate(x=x, y=y)

        distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)

        y_pred = self.estimator.predict(x=x)

        distance[np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)] = 0  # type: ignore
        predicted_class = np.where(distance > self.distance_threshold_tau, 1, 0)
        prob = np.zeros_like(distance)
        if self.threshold_bins:
            # bin accuracy is the probability of being a member
            for t_bin in self.threshold_bins:
                prob[distance > t_bin[0]] = t_bin[1]
        else:
            # use sigmoid on distance from threshold
            dist_threshold = distance - self.distance_threshold_tau
            prob = 1 / (1 + np.exp(-dist_threshold))
        return predicted_class, prob