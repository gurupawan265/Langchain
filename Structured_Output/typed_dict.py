from typing import TypedDict

class Person(TypedDict):

    name:str
    age:int

new_person:Person={'name':'pawan','age':15}

print(new_person)