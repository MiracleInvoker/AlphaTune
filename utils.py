import ast


def extract_datafields(alpha_expression):
    # 1. Sanitize: Make the string compatible with Python syntax parser
    # Replace &&/|| with and/or, and remove trailing semicolons if necessary
    alpha_expression = alpha_expression.replace("&&", " and ").replace("||", " or ")

    # 2. Parse the code into a tree
    tree = ast.parse(alpha_expression)

    # 3. Collect identifiers
    variables = set()
    assignments = set()
    operators = set()

    for node in ast.walk(tree):
        # Detect assignments
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.add(target.id)

        # Detect operators
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                operators.add(node.func.id)

        # Detect all variables
        elif isinstance(node, ast.Name):
            variables.add(node.id)

    # 4. Logic: Data Fields = Variables - (Operators + Assignments)
    datafields = variables - assignments - operators

    return sorted(datafields)
