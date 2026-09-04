Current training settings

The current FedProx setup uses a learning rate of (3e-4), weight decay of
(1e-4), one local epoch, and (mu=0.01). Training can run for up to 30 rounds and
stops if validation macro F1 does not improve for 10 rounds.

One local epoch had the best average validation result in the completed epoch
test. The learning rate and weight decay worked in the current runs but have
not been compared in a full tuning experiment.
