# CHARIS CAT 2025
# BABYLLM - babyLLM.py

import SCHOOL.staffroom.calligraphist as S_output
import SCHOOL.staffroom.counsellor as c
from config import *

councel = c.COUNSELLOR()
output = S_output.S_OUTPUT(councel)

for type, _ in output.S_types.items():
    print(output.S_apply(type, f"This is the {type} colour!"))

print(output.S_apply('superPerfect', "CHAWIS IS PERFECT uwu owo"))
print(output.S_apply('perfect', "ELODIE IS PERFECT owo uwu"))
#print(output.S_apply('almostPerfect', "BABYLLM IS ALMOST PERFECT"))
#print(output.S_apply('superGreat', "BABYLLM IS SUPER GREAT"))
#print(output.S_apply('great', "BABYLLM IS GREAT"))
#print(output.S_apply('good', "BABYLLM IS GOOD"))
#print(output.S_apply('fine', "BABYLLM IS FINE"))
#print(output.S_apply('almostFine', "CHARIS IS ALMOST FINE"))
#print(output.S_apply('average', "GEORGE IS AVERAGE"))
#print(output.S_apply('meh', "BABYLLM IS MEH"))
#print(output.S_apply('bad', "BABYLLM IS BAD"))
#print(output.S_apply('worse', "GEORGE IS WORSE"))
#print(output.S_apply('wtf', "KEVIN IS WTF"))
#print(output.S_apply('omg', "PETE IS OMG"))
#print(output.S_apply('omgwtf', "PETE IS OMGWTF"))
#print(output.S_apply('omgwtf', "CHARIS IS OMGWTF!"))
#print(output.S_apply('emergency', "BABYLLM IS EMERGENCY"))

print("❀ ʕ⊃✰✰⋆⋆")
print("✰✰⋆⋆")
print("✰✰⋆⋆꩜ ❀ ꩜ ʕ꩜ ⊃꩜")

print("ʕっ•̀o•́ʔっ✰✰⋆⋆")

print("ʕっ꩜ o꩜ ʔっ✰✰⋆⋆")

import inspect

def decorator(func):
    def inner(*args, **kwargs):
        caller_stack = []
        for stack in inspect.stack():
            caller_stack.append(stack[0].f_code.co_qualname)
        print(f"Calling {func.__class__}.{func.__name__} from: {', '.join(caller_stack)}")
        return func(*args, **kwargs)

    return inner

@decorator
def tryme1():
    pass

@decorator
def tryme2():
    tryme4()

@decorator
def tryme3():
    tryme2()

@decorator
def tryme4():
    tryme5()

@decorator
def tryme5():
    tryme1()
class t:
    def __init__(self):
        tryme3()

t()