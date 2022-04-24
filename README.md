# gym-tendulum
10th order pendulum on a cart to be used as a 3rd party gym environment

## Install
```bash
git clone git@github.com:jcranney/gym-tendulum.git
cd gym-tendulum
pip install -e .
```
To test that the install was successful, try running:
```bash
python test_tendulum.py
```

## Examples
### LQR
To run the `tendulum` example using [LQR](https://python-control.readthedocs.io/en/latest/generated/control.dlqr.html#control.dlqr) as the control law, navigate to `gym-tendulum/examples` and run:
```bash
python tendulum_lqr.py
```
With any luck, the tendulum will carefully navigate back to the origin. The system is very unstable, but the default parameters saved in the example file should be sufficient to control the tendulum.

You can experiment with different orders of n-dulum, but it's not super elegant at this stage. For now, you have to go into `gym-tendulum/envs/tendulum_env.py` and modify `self._n_order`:
```python
    def __init__(self):
        # order of the ndulum
        self._n_order = 10
```
to be some other number. Anything above 10 is not going to be stable without some tuning, below 10 should be OK, with 1 being the *most* stable.

Depending on how you installed `gym-tendulum`, you may need to reinstall after making this change. If you followed the instructions in the **Install** section, this should not be necessary.