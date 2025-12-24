import cplex
from docplex.mp.model import Model

# 1. Verify the Unlimited Engine
c = cplex.Cplex()
print(f"CPLEX Version: {c.get_version()}")

# 2. Verify the Library
mdl = Model(name='test')
mdl.continuous_var(name='x')
print("Model created successfully.")

def stress_test():
    print("--- STARTING STRESS TEST ---")
    mdl = Model(name='Stress_Test')
    
    # 1. Create 1,500 variables (Trial limit is 1,000)
    # If you are on the trial version, it might let you create them 
    # but will CRASH when you try to solve.
    print("Creating 1,500 variables...", end=" ")
    vars_list = mdl.continuous_var_list(1500, name='x')
    print("Done.")

    # 2. Add a massive constraint: Sum(x) <= 10000
    mdl.add_constraint(mdl.sum(vars_list) <= 10000)
    
    # 3. Maximize Sum(x)
    mdl.maximize(mdl.sum(vars_list))
    
    # 4. Attempt to Solve
    print("Attempting to solve big model...")
    s = mdl.solve()
    
    if s:
        print("\n SUCCESS! Model solved.")
        print("You definitively have the UNLIMITED version.")
        print(f"Objective Value: {s.objective_value}")
    else:
        print("\n FAILED. Model did not solve.")
        # If it fails here with a specific CPLEX error about "size limit", 
        # then the license is not working.

if __name__ == "__main__":
    stress_test()