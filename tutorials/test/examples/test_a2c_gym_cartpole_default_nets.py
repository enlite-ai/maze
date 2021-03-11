"""Contains unit tests for examples."""
from examples.a2c_gym_cartpole_default_nets import main


def test_a2c_gym_cartpole_ff():
    """ unit tests """
    main(n_epochs=1, rnn_steps=0)


def test_a2c_gym_cartpole_rnn():
    """ unit tests """
    main(n_epochs=1, rnn_steps=5)
