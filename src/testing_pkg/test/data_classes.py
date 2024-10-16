from enum import Enum


class TestType(int, Enum):
    """
    TestType is an enumeration that inherits from int and Enum. It represents different types of tests:

    Attributes:
        type0 (int):
            Represents an undefined type with a value of 0.
        type_sub (int):
            Represents substitution test, against incorrect synonym.
        type_relnc (int):
            Represents test with prompt where {src} and {word} can be replaced, like relation w/o context.
        type_relc (int):
            Represents test with prompt wh. {example}, {src} and {word} can be replaced, like relation w/ context.
        type_ref (int):
            Represents test with prompt wh. {example} and {word} can be replaced, like reference w/ context.
        type_relnc_rand (int):
            Represents test with prompt where {src} and {word} can be replaced, like relation w/o context.
            Negative examples are not only the incorrect synonym but also random words.
        type_relc_rand (int):
            Represents test with prompt wh. {example}, {src} and {word} can be replaced, like relation w/ context.
            Negative examples are not only the incorrect synonym but also random words.
        type_sub_rand (int):
            Represents substitution test, against incorrect synonym and random words.
    """
    type0 = 0  # undefined
    type_sub = 1
    type_relnc = 2
    type_relc = 3
    type_ref = 4
    type_relnc_rand = 5
    type_relc_rand = 6
    type_sub_rand = 7


class TestElements:
    """
    TestElements represents a construction made out of a triple (src, synonym, other) and an example,
    as well as the index of the src in the example.

    Attributes:
        src (str): The source string.
        synonym (str): The synonym string.
        other (str): Another synonym string.
        index (int): The index of the source in the example.
        example (str): The example string.

    Methods:
        __init__(tlist):
            Initializes the TripleExample with a list of attributes.
    """
    def __init__(self, tlist):
        self.src = tlist[0]
        self.synonym = tlist[1]
        self.other = tlist[2]
        self.index = int(tlist[3])
        self.example = tlist[4]


class TestRecord:
    """
    Represents a test item, which can be formed of either:
    - A single item number, flag, word, and context.
    - A list or tuple containing four elements: [itemnb, flag, word, context].

    Attributes:
        itemnb (int): The item number.
        flag (str): The flag.
        word (str): The word.
        context (str): The context.

    Methods:
        __init__(*args):
            Initializes the instance with provided arguments.
    """
    def __init__(self, *args):
        """
        Initializes the instance with provided arguments.

        Args:
            *args: Variable length argument list. Can be either:
                - Four individual arguments: itemnb, flag, word, context.
                - A single argument which is a list or tuple containing four elements: [itemnb, flag, word, context].

        Raises:
            ValueError: If the number of arguments is not 1 or 4.
        """
        if len(args) == 4:
            self.itemnb = args[0]
            self.flag = args[1]
            self.word = args[2]
            self.context = args[3]
        elif len(args) == 1:
            self.itemnb = args[0][0]
            self.flag = args[0][1]
            self.word = args[0][2]
            self.context = args[0][3]
        else:
            raise ValueError("Wrong number of arguments")

    def __str__(self):
        return f'{self.itemnb}\t{self.flag}\t{self.word}\t{self.context}'


class Flag(Enum):
    """
    Flag is an enumeration that represents different types of flags with associated integer values.
    The flags represent the type of sentence in the test:

    SOURCE (reference sentence),
    SYNONYM (sentence target word replaced with a synonym), and OTHER (sentence with another word).

    Attributes:
        SOURCE (int): Represents the source flag with a value of 2.
        SYNONYM (int): Represents the synonym flag with a value of 1.
        OTHER (int): Represents the other flag with a value of 0.
    Methods:
        list() -> list:
            Returns a list of the integer values of the enumeration members.
    """
    SOURCE = 2
    SYNONYM = 1
    OTHER = 0

    @staticmethod
    def list():
        return list(map(lambda c: c.value, Flag))
