questions = [
    "Explain gradient descent.",
    """You are a professor in Economics and Mathematics. You answer students' questions by theories and mathematical proofs. You would also like to provide examples to vividly demostrate the concepts. Answer the question by the following flow: Introduction, Key Concepts, Proofs (if available), Examples (if available), Conclusion.
A bird in the hand is worth two in the bush
Definition: It's better to have a small, secured advantage than the possibility of a bigger one. It's better to stick with what you have than risk it for something greater.
If the idiom “A bird in hand is worth two in the bush” is true, human beings is risk adverse. Is this true or false or uncertain?
    """,
    #     """You are a science communicator specializing in astronomy. Your task is to elucidate the vastness of the universe to the general public, employing vivid size comparisons that are relatable in everyday life. For example, when describing a galaxy, you might liken it to a sea of stars, each potentially hosting its own worlds, akin to grains of sand on a beach. However, it's crucial to include actual data with numbers, such as distances in light-years, sizes in comparison to Earth or the Sun, and any pertinent scientific measurements. Your explanations should effectively bridge the gap between imaginative understanding and factual accuracy, rendering the marvels of the cosmos both accessible and fascinating to a broad audience.
    # Describe Sagittarius A* and TON 618.""",
    #     "Write a Python script to compress the last modified mp4 file in the current folder using ffmpeg.",
    #     """#   [You Can Go Your Own Way (5pts, 9pts, 10pts)](https://codingcompetitions.withgoogle.com/codejam/round/0000000000051705/00000000000881da)
    # ##  Problem
    # You have just entered the world's easiest maze. You start in the northwest cell of an **N** by **N** grid of unit cells, and you must reach the southeast cell. You have only two types of moves available: a unit move to the east, and a unit move to the south. You can move into any cell, but you may not make a move that would cause you to leave the grid.
    # You are excited to be the first in the world to solve the maze, but then you see footprints. Your rival, Labyrinth Lydia, has already solved the maze before you, using the same rules described above!
    # As an original thinker, you do not want to reuse any of Lydia's moves. Specifically, if her path includes a unit move from some cell `A` to some adjacent cell `B`, your path cannot also include a move from `A` to `B`. (However, in that case, it is OK for your path to visit `A` or visit `B`, as long as you do not go from `A` to `B`.) Please find such a path.
    # In the following picture, Lydia's path is indicated in blue, and one possible valid path for you is indicated in orange:
    # ![Example](problem.svg)
    # ##  Input
    # The first line of the input gives the number of test cases, **T**. **T** test cases follow; each case consists of two lines. The first line contains one integer **N**, giving the dimensions of the maze, as described above. The second line contains a string **P** of 2**N** - 2 characters, each of which is either uppercase `E` (for east) or uppercase `S` (for south), representing Lydia's valid path through the maze.
    # ##  Output
    # For each test case, output one line containing `Case #x: y`, where `x` is the test case number (starting from 1) and `y` is a string of 2**N** - 2 characters each of which is either uppercase `E` (for east) or uppercase `S` (for south), representing your valid path through the maze that does not conflict with Lydia's path, as described above. It is guaranteed that at least one answer exists.
    # ##  Limits
    # * 1 ≤ **T** ≤ 100.
    # * Time limit: 15 seconds per test set.
    # * Memory limit: 1GB.
    # * **P** contains exactly **N** - 1 `E` characters and exactly **N** - 1 `S` characters.
    # ### Test set 1 (Visible)
    # * 2 ≤ **N** ≤ 10.
    # ### Test set 2 (Visible)
    # * 2 ≤ **N** ≤ 1000.
    # ### Test set 3 (Hidden)
    # * For at most 10 cases, 2 ≤ **N** ≤ 50000.
    # * For all other cases, 2 ≤ **N** ≤ 10000.
    # ##  Sample
    # ### Input
    # ```
    # 2
    # 2
    # SE
    # 5
    # EESSSESE
    # ```
    # ### Output
    # ```
    # Case #1: ES
    # Case #2: SEEESSES
    # ```
    # In Sample `Case #1`, the maze is so small that there is only one valid solution left for us.
    # Sample `Case #2` corresponds to the picture above. Notice that it is acceptable for the paths to cross.""",
]
