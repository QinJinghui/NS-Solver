# NS-Solver

## Training
```
1) cd language_models

2) edit train_language_model_train_test.py and set train_for_question=True

3) run lm for question: python3 train_language_model_train_test.py 

4) edit train_language_model_train_test.py and set train_for_question=False

5) run lm for equation: python3 train_language_model_train_test.py

6) copy the lm checkpoints in the solver save dir and then run the solver:
    6.1) cd solver
    6.2) python3 run_ns-solver.py
```
