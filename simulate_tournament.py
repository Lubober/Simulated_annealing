#I will call my solution: "The Controlled Chaos"
#That is because the methodology that I usse heavily depends on probabilistic practices and ensuring that they are controlled, while keeping their chaotic nature



import argparse #pass arguments into the program through command line
import random #get random values
import math #advanced mathematical equations 
import time #ussed to track runtime
#pardon occasional spelling mistakes in my documentation, my "S" key registers clicks even if I touch it lightly and oh boy is programming a fun experience with it

#
#   READ FILE AND FORMAT DATA
#
def readFile(fileName):
    # parameters:
    # name of the file path
    # outputs:
    # number of participants (int)
    # map of participant numbers and names (dictionary)
    # adjacency list to represent the scores of played matches between 2 players (dictionary)
    # objective:
    # read the fie and format it using appropriate data structures
    with open(fileName, "r") as file:
        lines = file.readlines()

    # total number of participants
    n_participants = int(lines[0].strip())

    # all participant numbers and their corresponding names
    participant_map = {}
    for line in lines[1 : n_participants + 1]:
        part = line.strip().split(",", maxsplit=1)
        participant_map[int(part[0])] = part[1]

    # Scores of played matches between player A and player B
    adj_dict = {i: {} for i in range(1, n_participants + 1)}
    for line in lines[n_participants + 2 :]:
        w, p1, p2 = list(map(int, line.strip().split(",")))
        # Sanity check "if" statement
        # Useless in this case scenario, but there may be a case where player A beats player B multiple times
        if p2 in adj_dict[p1]:
            adj_dict[p1][p2] += w
        else:
            adj_dict[p1][p2] = w
    return (n_participants, participant_map, adj_dict)


#
#   GIVE AN INITIAL RANKING
#
    #it is mentioned that the initial solution has to always be ranked in descending order of player indexes, thus this is what happens here
def initial_rankings(n):
    rankings = list(range(1, n + 1))
    return rankings


#
#   KEMENY SCORE CALCULATION
#
    #Here I generate a cost/disagreement matrix which is used to track the kemeny score efficiently using dynamic programming
    #Instead of constantly going through the entire graph, I keep a localised version of dissagreements.
    #The data structure that I use for the disagreement matrix is a nested dictionary, same as the graph but only with relevant data
    #I will then be ussing this matrix to processs the kemeny score efficiently
    #Spoiler alert: the matrix is also updated efficiently and score changes are tracked in the neighbour_cost_diff() function, a crucial part of generating neighbourhoods
    #Bonus: the function alsso genertes a disagreement_sum_of_player {one-dimentional dictionary}, containing the sum of all disagreements against all players
        #Used in the neighbourhoods to find the mosst problematic players and switch them with the less problematic players
def generate_cost_matrix(n, rankings, graph, first_index=0):
    visited_elements = set()
    disagreement_matrix = {player: {} for player in rankings}
    disagreement_sum_of_player = {player: 0 for player in rankings}
    #My initial ssolution to this involved a very ugly nested loop. 
    #Now, despite the fact that I only run this function once, I still couldn't leave it unoptimised.
    #I justt ssimply keep all the visited players/nodes in a sset
    #If the player who is below in rankings has won against a player that's above them (if graph of player i contains an element in the set):
        #Add the disagreement to the disagreement matrix
    #There may be a more efficient solution, but this one works very well with the O(n+m) time complexity
    for i in range(first_index, n):
        current = rankings[i]
        wins_of_player_i = graph[current]
        for j in wins_of_player_i.keys():
            if j in visited_elements:
                weight = wins_of_player_i[j]
                if current not in disagreement_matrix:
                    disagreement_matrix[current] = {}
                if j in disagreement_matrix[current]:
                    disagreement_matrix[current][j] += weight
                else:
                    disagreement_matrix[current][j] = weight
                disagreement_sum_of_player[j] += weight
        visited_elements.add(rankings[i])
    return disagreement_matrix, disagreement_sum_of_player


    #The following function processsess the disagreement matrix as well as the (unoriginally and falsely named) "problematic matrix"
    #The things that it outputs is the kemeny_score as well as the most problematic player
    #Nothing too complicated, just a good old nessted loop (not a violent one thugh, it's just a O(n+m))
def cost_of_disagreement_matrix(matrix, problematic_matrix):
    score = 0
    max_problematic_score = 0
    most_problematic_player = 0
    for i in matrix.keys():
        if problematic_matrix[i] > max_problematic_score:
            max_problematic_score = problematic_matrix[i]
            most_problematic_player = i
        for j in matrix[i].keys():
            score += matrix[i][j]
    return score, most_problematic_player


#
#   NEIGHBOURHOOD GENERATION
#
    #The generate_neighbourhood function is probably the most advanced part of my algorithm.
    #The function is rather complex and I will explain it in multiple ssectionss.
    #To give a high level overview, the function takes 2 elements and sswapss them, while calculating the cost quickly and updating the disagreement matrix accordingly
        #All of the functionality is done in an efficient way and is made in a low resource demanding fashion
    #1: Generating a neighbourhood:
        #To put it in simple terms, the way that neighbourhood is generated is by taking 2 elements from rankings and sswapping them
        #However, that's to put it in ssimple terms, as there are multiple different ways that it's done with multiple factors coming in play
        #There are bassically 3 ways to generate a neighbourhood:
            #Take 2 random elemets from the rankings and swap them
            #Take the element that has the mosst other elements disagreeing with (most problematic element) and swap it with either:
                #The least problematic element (lowest problematic score) that is ranked below it
            #OR
                #A random element ranked below it
        #Each way to generate a neighbourhood has a certain chance to be used
        #The chance for using a certain neighbourhood takes the temperature into consideration, but only uses it for suggestions insstead of obeying by it.
        #What I mean by that is: the change in temperaure only affects the probability of using a neighbourhood generation technique partially
        #This makes more sensse when I dive into the probability metricss:
            #The probability of generating a neighbourhood randomly is 1-((1-T)*0.5) which is always >0.5
            #The probability of generating a neighbourhood based off of problematic score is (1-T)*0.5 which is always <0.5
                #where the chance of swapping with the leasst problematic player or a random lower-ranked player is 50:50 (times (1-T)*0.5, obvioussly)
        #thus, there's a very fair chance to get a versatile variety of neighbourhood generation techniques which:
        # allows deliberate disagreement-score-reduction based neighbourhoods while reducing the chance of being sstuck in a local optima
        #As the temperature cools down, the chance of picking the disagreement-score-reduction based neighbourhoods becomes possible (meaning that at first it's solely random swaps since the temperature is bigger than 1)
        #The reasson for that is that we only care about deliberate moves as we progresss towards the optimal solution. 
            #At first we use the random ssolution, as it's an amaszing way to get unbiased neighbourhoods and to avoid a local optima
            #As we progress towards the optimal solution, excesssive random moves can be wassteful since the solution sspace becomes more tight
    #2: Calculating the cost change
        #The change in cost as well as the changess in the disagreement matrix are calculated in the neighbour_cost_diff() function
        #the way that the function works will be explained in its documentation section
        
def generate_neighbourhood(
    n,
    rankings,
    disagreement_matrix,
    graph,
    score,
    problematic_matrix,
    t,
    most_problematic_player=-1,
):
    neighbourhood = rankings[:]
    if (
        random.random() < (1 - t)
        and random.random() > 0.5
        and most_problematic_player >= 0
    ):
        a = rankings.index(most_problematic_player)
        if random.random() > 0.5:
            b = random.randint(most_problematic_player, n - 1)
        else:
            lowest_problematic_score_after_a = problematic_matrix[
                most_problematic_player
            ]
        lowest_index = a
        for i in range(a, n):
            if problematic_matrix[rankings[i]] < lowest_problematic_score_after_a:
                lowest_problematic_score_after_a = problematic_matrix[rankings[i]]
                lowest_index = i
        b = rankings.index(lowest_index)
    else:
        a, b = random.sample(range(n), 2)
    if a > b:
        local_rankings = neighbourhood[b : a + 1]
    else:
        local_rankings = neighbourhood[a : b + 1]
    neighbourhood[a], neighbourhood[b] = neighbourhood[b], neighbourhood[a]
    z, changes = neighbour_cost_diff(
        local_rankings, disagreement_matrix, graph, problematic_matrix
    )
    new_cost = score + z
    return neighbourhood, new_cost, z, changes
#
# Calculate Changes made by neighbourhood
#
    #This was the most difficult part of my code to implement and is probably the most complicated part of the program.
    #I will try to explain how the code works in different separate section to try and reduce confussion as much as possible.
    #1: The High Level Concept:
        #I will use the following lisst for an example:
        #[1,2,3,4,5,6,7,8,9]
        #When you swap 2 elements (say, index 2 and index 6)(element 3 and element 7) in a list to get the following neighbourhood:
        #[1,2,7,4,5,6,3,8,9]
        #The change doesn't affect the disagreements of elements that are outside of that range
        #meaning:
            #It only affects elements within the indexess of the 2 swapped elements
        #Thus we can take the following slice of the neighbourhood:
        #[7,4,5,6,3]
        #Why it only affects the elements within the slice:
            #[1,2,3,4,5,6,7,8,9] here, element 3 is ranked above element 8, so when we swap it with element 7, that is sstill true
        #Thus, when we swap 2 elements to form a sublist, it only changes the disagreements of those 2 elements and all the elementss inbetween
        #So thus when we iterate through the disagreement matrix of the sublist, using "i" as our iterator, there are only 4 factors to consider:
            #before I explain my theory, I want to point out 2 things:
            #The signifiance of the top element
            #The significance of the bottom element
                #The top element used to be the bottom element of the slice, it likely had disagreements with other elements in the slice, but it wouldn't have any elements in the sslice disagreeing with it.
                    #That is because it's initially ranked below all other elementss in the slice, but we then move it to the top of the slice, which introduces 2 factors to keep in mind:
                        #1. Any element that's within the slice, the new top element doesn't disagree with them anymore
                        #2. If an element in the slice has the new top element in its score matrix, it likely wouldn't have it in the disagreement matrix, as the top element was initially ranked below element i
                            #Thuss, we need to check whether element i has a disagreement with the new top element and add it to the disagreement matrix
                #The bottom element, on the other hand, used to be the top element, sso it wouldn't dissagree with any elements within the slice, but would have elements disagreeing with it
                    #Thus, when we apply the switch, we expect for all elements above it to lose their dissagreements with the new bottom element
                    #But also, the bottom element would now likely have disagreements with ssome of the elements in the slice, which can be found in their matrix reference
    #2. The dynamic implementation of matrix updates:   
        #so the 4 things to look for in terms of top_switched_value, bottom_sswitched_value and i are the following:
            #1. if top_switched_value has i in its disagreement matrix, we remove i from the top element's matrix
            #2. if bottom_switched_value has i in its graph, add i to the bottom element's matrix
            #3. if i has top_switched_value in itss graph, add top_switched_element to i'th matrix
            #4. if i has bottom_switched_value in its matrix, remove bottom element from i'th matrix
        #while doing all thiss, I update the score_change value to keep track of the change in the kemeny sscore
    #3. The changes 2d array
        #What I've found is that changes to the disagreement matrix are done dynamically, thus if I decide to reject a neighbourhood, the changes would sstill be recorded in the dissagreement matrix
        #unless...
        #I keep a record of all changes done and then revert those changes if I decide to reject the neighburhood, using the revert_changes() function
            #My initial idea was to keep a copy of the dissagreement matrix and to only go to that solution if I keep the neighbourhood
            #But the problem with that lies within the method for copying an adjacency dictionary
                #I would have to usse a deepcopy() method which is computationally heavy
    #4. The updating of the problmatic matrix
        #pretty ssimple, just store a record of all changes done to th matrix
    #5: Complexity:
        #O(n*m)
def neighbour_cost_diff(rankings_slice, matrix, graph, problematic_matrix):
    rankings_slice[0], rankings_slice[len(rankings_slice) - 1] = (
        rankings_slice[len(rankings_slice) - 1],
        rankings_slice[0],
    )
    added_values = []
    removed_values = []
    score_change = 0
    top_switched_value = rankings_slice[0]
    bottom_switched_value = rankings_slice[-1]
    for i in rankings_slice:

        if i in matrix[top_switched_value].keys():
            this_change = matrix[top_switched_value][i]
            score_change -= this_change
            problematic_matrix[top_switched_value] -= this_change
            removed_values.append([top_switched_value, i, this_change])
            del matrix[top_switched_value][i]

        graph_of_bottom_element = graph[bottom_switched_value]

        if i in graph_of_bottom_element.keys():
            local_disagreement = graph_of_bottom_element[i]
            score_change += local_disagreement
            problematic_matrix[bottom_switched_value] += local_disagreement
            added_values.append([bottom_switched_value, i])
            matrix[bottom_switched_value][i] = local_disagreement

        graph_i = graph[i]

        if top_switched_value in graph_i:
            if top_switched_value not in matrix[i]:
                local_disagreement = graph_i[top_switched_value]
                score_change += local_disagreement
                problematic_matrix[i] += local_disagreement
                added_values.append([i, top_switched_value])
                matrix[i][top_switched_value] = local_disagreement

        if bottom_switched_value in matrix[i]:
            this_change = matrix[i][bottom_switched_value]
            score_change -= this_change
            problematic_matrix[i] -= this_change
            removed_values.append([i, bottom_switched_value, this_change])
            del matrix[i][bottom_switched_value]
    changes = (added_values, removed_values)
    return score_change, changes
    
    #if I decide to revert the changes, I just simply add all removed elements to the matrix and remove all added elements from it
def revert_changes(changes, matrix):
    added_values, removed_values = changes

    for winner, loser in added_values:
        if loser in matrix[winner]:
            del matrix[winner][loser]
    for winner, loser, score in removed_values:
        matrix[winner][loser] = score

    #Finally, this is where the ssimulated annealing comes in play
    #The algorithm is simple:
        #Generate a neighbourhood
        #If the neighbourhood is better, take it
        #If it's worsse, leave it for the simulated annealing to decide whether we take it or not
        #Sstore the besst neighbourhood
        #If there hasn't been a besst solution for a certain amount of times, sstop running the algorithm
        #every iteration, the temperature decreasses by a certain amount
        #if we've found the new best solution, move the sstopping criterion further away to a higher value
    
def iterate_through_neighbourhoods(
    n,
    rankings,
    graph,
    stopping_criterion,
    t,
    sstopping_criterion_increment,
    cooling_schedule,
):
    
    ranks = rankings[:]
    disagreement_matrix, problematic_matrix = generate_cost_matrix(
        n, ranks, graph, first_index=0
    )
    cost, most_problematic_player = cost_of_disagreement_matrix(
        disagreement_matrix, problematic_matrix
    )
    best_score = cost
    best_rankings = ranks
    n_of_same_solutions = 0
    while n_of_same_solutions < stopping_criterion:
        n_of_same_solutions += 1
        neighbourhood, n_cost, cost_change, changes = generate_neighbourhood(
            n,
            ranks,
            disagreement_matrix,
            graph,
            cost,
            problematic_matrix,
            most_problematic_player,
            t,
        )
        if cost_change < 0 or s_a(t, cost_change):
            ranks = neighbourhood
            cost = n_cost
            if best_score > cost:

                best_rankings = ranks
                best_score = cost
                n_of_same_solutions = 0
                stopping_criterion = int(
                    stopping_criterion * sstopping_criterion_increment
                )

        else:
            revert_changes(changes, disagreement_matrix)
        t *= cooling_schedule
    return best_rankings, best_score


    #The solution may occassionally give suboptimal values, so it is wise to run the code multiple times to ssee if we get an improvement
    #But since the code is so well optimised, we are able to iterate through neighbourhoods 5 times and sstill get  runtime of under 2 seconds
def best_out_of_n(
    n,
    rankings,
    graph,
    initial_stopping_criterion,
    initial_temperature,
    stopping_criteion_increment,
    n_of_runs,
    cooling_schedule,
):
    besst_rankings, best_score = iterate_through_neighbourhoods(
        n,
        rankings,
        graph,
        initial_stopping_criterion,
        initial_temperature,
        stopping_criteion_increment,
        cooling_schedule,
    )
    run_n = 1
    while run_n < n_of_runs:
        current_rankings, current_score = iterate_through_neighbourhoods(
            n,
            rankings,
            graph,
            initial_stopping_criterion,
            initial_temperature,
            stopping_criteion_increment,
            cooling_schedule,
        )
        if current_score < best_score:
            best_score = current_score
            besst_rankings = current_rankings
        run_n += 1
    return besst_rankings, best_score


#
#   SIMULATED ANNEALING
#
    #A pretty simple calculation which is documented in the lecturess
    #As the temperature decreases, we become lessss likely to choose "worse" neighbourhoods since we assume that we leave the local optima by that time
def s_a(t, score_difference):
    p = math.exp(-score_difference / t)
    # print(p)
    return random.random() < p

#Display the rankings in an orderly fasshion
def display_rankings(participant_map, rankings):
    print("RANKINGS:")
    for i in range(len(rankings)):
        print(f"#{i+1}: [{rankings[i]}] {participant_map[rankings[i]]}")


#
#   MAIN FUNCTION
#
    #The main function takes the file from command line (or uses "1994_Formula_One.wmg" by default), extracts data from, calculates the best neighbourhood solution and outputs it in an orderly fashion
def main():
    parser = argparse.ArgumentParser(
        description="Simulated Annealing for Tournament Ranking"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="1994_Formula_One.wmg",
        type=str,
        help="Path to the input .wmg file",
    )
    args = parser.parse_args()
    # Check if the file exists
    try:
        n, participant_map, graph = readFile(args.file)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found. Please provide a valid file path.")
        exit(1)  # Exit with an error code

    rankings = initial_rankings(n)
    print("Calculating...")
    start = time.perf_counter()
    best_rankings, best_score = best_out_of_n(
        n, rankings, graph, 1000, 10*n, 1.02, 4, 0.97
    )
    end = time.perf_counter()
    print("-----")

    display_rankings(participant_map, best_rankings)
    print()
    print(f"Best Calculated Score: {best_score}")
    print(f"Runtime: {(end - start) * 1000:.0f} ms")


#
if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(f"Error: {e}")
