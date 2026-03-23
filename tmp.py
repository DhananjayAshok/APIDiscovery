def validate_input_args(*args, **kwargs) -> None:
    pass

def test_func(arg0, arg1):
    validate_input_args(arg0, arg1)
    replace_with = arg1
    self = arg0

    new_string = ""
    for char in self:
        if char == " ":
            new_string += replace_with
        else:
            new_string += char
    return new_string

test_examples = [["(\"test string\", \"\")", "teststring"], ["(\"no spaces\", \"*\")", "no*spaces"], ["(\"hello world\", \"-\")", "hello-world"], ["(\"space here !\", \"@\")", "space@here@!"], ["(\" \", \".\")", "..."], ["(\"this is a test\", \"_\")", "this_is_a_test"], ["(\"hello@world\", \"&\")", "hello@world"], ["(\"a b\", \"#\")", "a##b"]]
for example in test_examples:
    input_args, expected_output = example
    result = test_func(*eval(input_args))

