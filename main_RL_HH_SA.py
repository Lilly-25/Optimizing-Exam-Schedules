import csv
import collections
import math
import time
import random as rn
from numpy import concatenate
from numpy import random
from numpy.random import randint
import copy

days = [("05-FEB-2024","Monday", 0), ("06-FEB-2024","Tuesday", 1), ("07-FEB-2024","Wednesday", 2), ("08-FEB-2024","Thursday", 3), ("09-FEB-2024","Friday", 4),("10-FEB-2024","Saturday", 5),("11-FEB-2024","Sunday", 6),
        ("12-FEB-2024","Monday", 7), ("13-FEB-2024","Tuesday", 8), ("14-FEB-2024","Wednesday", 9), ("15-FEB-2024","Thursday", 10), ("16-FEB-2024","Friday", 11),("17-FEB-2024","Saturday", 12),("18-FEB-2024","Sunday", 13),
        ("19-FEB-2024","Monday", 14), ("20-FEB-2024","Tuesday", 15), ("21-FEB-2024","Wednesday", 16), ("22-FEB-2024","Thursday", 17), ("23-FEB-2024","Friday", 18)]
totalDays = len(days)
examStartTiming = [(8.5,0),(11,1),(13.5,2),(16,3)]
totalExamStartTiming = len(examStartTiming)
examDuration = 1.5
courses =[]
instructors = []
classRooms =[]
registrations = []
totalInstructors = -1
totalClassRoom = -1
max_count = []


# Class to store course

class Course:
    
    # Initialize the class
    def __init__(self,code,name,number):
        self.courseCode = code
        self.courseName = name
        self.number = number
    
    # Print Course
    def __repr__(self):
        return '({0},{1},{2})'.format(self.courseCode, self.courseName,self.number)
    
    # Check Equality 
    def __eq__(self,other):
        return self.courseName==other.courseName and self.courseCode==other.courseCode
    
# Class to store Class Room Capacity

class Room:
    
    #Initialize the class
    def __init__(self,roomNo,capacity,index):
        self.roomNumber = roomNo
        self.capacity = capacity
        self.index = index
        
    #Print Room
    def __repr__(self):
        return '({0},{1},{2})'.format(self.roomNumber,self.capacity,self.index)
    
    #Check Equality
    def __eq__(self, other):
        return self.roomNumber == other.roomNumber and self.capacity == other.capacity
  
# Class to store student registered in course

class Registration:
    
    # Initialize the class
    def __init__(self,courseCode,matrNumbers):
        
        self.registeredCourse = courseCode
        self.studentIds = matrNumbers.copy()
    
    #Print registration
    def __repr__(self):
        return '({0},{1})'.format(self.registeredCourse,self.studentIds)
    
    #Check equality
    def __eq__(self,other):
        if self.registeredCourse == other.registeredCourse and len(self.studentIds)==len(other.studentIds):
            count = 0
            for i in range(len(self.studentIds)):
                if self.studentIds[i] == other.studentIds[i]:
                    count +=1
            if count == len(self.registeredCourse):
                return True
        return False

#Class to store an exam
class Exam:
    
    #Initialize the class
    def __init__(self,course,startTime,roomNo, day, invigilator):
        self.course = course
        self.startTime = startTime
        self.roomNo = roomNo.copy()
        self.day = day
        self.invigilator = invigilator.copy()
        self.binary = []
    
    #Print an Exam
    def __repr__(self):
        return '(\n {0},{1},{2},{3},\n{4}'.format(self.course.courseName,self.startTime, self.roomNo,self.day,self.invigilator)
    
    #Check equality
    def __eq__(self,other):
        if self.courseCode == other.courseCode and len(self.roomNo) == len(other.roomNo) and self.startTime == other.startTime and len(self.invigilator) == len(other.invigilator):
            count = 0
            count1 = 0
            for i in range(len(self.roomNo)):
               if self.roomNo[i] == other.roomNo[i]:
                   count +=count1
            for i in range(len(self.invigilator)):
                if self.invigilator[i] == other.invigilator[i]:
                    count1 +=1
                    
# Reading from files
def takeInput():
    # Reading courses from file
    with open('course.csv') as file:
        reader = csv.reader(file)
        count = 0
        for row in reader:
            if len(row) !=0:
                temp_course = Course(row[0],row[1],count)
                if temp_course not in courses:
                    courses.append(temp_course)
                    count +=1 
                    
    #Reading Room Details from file
    with open('RoomCapacity.csv') as file:
        reader = csv.reader(file)
        next(reader,None)
        count = 0
        for row in reader:
            if len(row) !=0:
                temp_room = Room(row[1],row[2],count)
                if temp_room not in classRooms:
                    classRooms.append(temp_room)
                    count +=1
    global totalClassRoom
    totalClassRoom = len(classRooms)
                
                    
    # Reading instructors from file
    
    with open('teachers.csv') as file:
        reader = csv.reader(file)
        count = 0
        for row in reader:
            if len(row) !=0:
                if row[0] not in instructors:
                    instructors.append((row[0],count))
                    count+=1
    global totalInstructors
    totalInstructors = len(instructors)
    
    # Reading Exam Registrations of student from file
    
    with open('ExamRegistration.csv') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            flag = False
            checkCourseCode = [x for x in courses if x.courseCode == row[1]]
            if len(row) !=0 and row[0] != '' and row[1] !='' and len(checkCourseCode):
                temp_registration = Registration(row[1],[row[0]])
                if len(registrations) != 0:
                    for i in registrations:
                        if i.registeredCourse == row[1] and row[0] not in i.studentIds:
                            i.studentIds.append(row[0])
                            flag = True
                            break
                if not flag:
                    registrations.append(temp_registration)
    # Getting Top 5 registered courses
    registeredCourse_count = [(r.registeredCourse,len(r.studentIds)) for r in registrations]
    sortedCourse_count = sorted(registeredCourse_count, key= lambda x:x[1],reverse=True)
    max_count.extend([r[0] for r in sortedCourse_count][:5])
    # print(max_count)
    # print(zip(*registeredCourse_count))
    range_label = ['0-50','50-100','100-200']
    reg_count = [0,0,0]
                    
    # Creating key-value pair of course and student registrations count
    # for r in registrations:
    #     student_registrations_count[r.registeredCourse] = len(r.studentIds)
    
                    
    
# Genrating random exam
def getRandomExam(course):
    courseCode = course.courseCode
    total_students = -1
    
    for x in registrations:
        if courseCode == x.registeredCourse:
            total_students = len(x.studentIds)
            break
    print(course.courseName, total_students)
    #Assinging rooms required to accomodata all students
    roomNo = []
    while total_students > 0:
      temp = classRooms[rn.randrange(0,totalClassRoom)]
      while temp in roomNo:
          temp = classRooms[rn.randrange(0,totalClassRoom)]
      roomNo.append(temp)
      total_students -= int(temp.capacity)
    # print(roomNo)
    #Setting time
    startTime = examStartTiming[rn.randrange(0,totalExamStartTiming)]
    
    #Setting Day
    day = days[rn.randrange(0,totalDays)]
    
    #Assiging invigilator required to invigilate all rooms
    invigilator = []
    for i in range(len(roomNo)):
        temp = instructors[rn.randrange(0,totalInstructors)]
        while temp in invigilator:
            temp = instructors[rn.randrange(0,totalInstructors)]
        invigilator.append(temp)
    # print(invigilator)
    return Exam(course,startTime,roomNo,day,invigilator)   

#Generate Random Initial Solution
def generate_init_solution():
    
    timetable = []
    for j in courses:
        timetable.append(getRandomExam(j))
    return timetable
    

takeInput()
# for r in classRooms:
#     print(r)
# print(totalClassRoom)

# for i in instructors:
#     print(i)
# print(totalInstructors)

for e in registrations:
    print(e)
print(len(registrations))





def select_heuristic(iteration, llhList):
    # Calculate epsilon decay
    # print(iteration)
    epsilon_decay = 1 / math.sqrt(iteration)
    index = -1
    
    # Initialize a random number generator for double values
    rand_double = random.random()
    # print(rand_double,epsilon_decay)
    if rand_double < epsilon_decay:
        # Under epsilon decay probability, do random selection
        # print('random')
        index = random.randint(0, len(llhList) - 1)
    else:
        # Select the index with the maximum utility value
        # print('max')
        max_value = max(llhList)
        
        # Randomly select one if there are multiple low-level heuristics with the same utility value
        index = random_duplicate_selection_sum(max_value, llhList)
    
    return index

# def llh_max_utility_index(llhList):
#     # Check if llhList is empty
#     if not llhList:
#         return None  # Return None for an empty list

#     # Find the index with the maximum utility value
#     max_index = 

#     return max_index

def random_duplicate_selection_sum(max_value, llhList):

    # Find all indices with the same utility value as llhList[index]
    duplicates = [i for i, value in enumerate(llhList) if value == max_value]

    # Randomly select one index from the duplicates
    selected_index = random.choice(duplicates)

    return selected_index

def quality(current_timetable):
    hard_constraints_violations= calculate_hard_constraints(current_timetable)[0]
    soft_constraints_violations = calculate_soft_constraints(current_timetable)[0]
    #print('Hard Constraints Violations',hard_constraints_violations)
    #print('Soft Constraints Violation',soft_constraints_violations)
    total_penalty = hard_constraints_violations + soft_constraints_violations
    # print('Total Penalty',total_penalty)
    return total_penalty

def calculate_hard_constraints(chromosome):
    
    hard_constraints_violations = 0
    h1 = True
    h2 = True
    h3 = True
    h4 = True
    h5 = True
    
    # H1: A student cannot sit more than more one exam at the same time.
    for exam in chromosome:
        
        students_exam1 = set([el for s in registrations if s.registeredCourse == exam.course.courseCode for el in s.studentIds])
        # print('student1',students_exam1)
        exam_on_sametime = [e for e in chromosome if e.day == exam.day and e.startTime == exam.startTime and e.course.courseCode != exam.course.courseCode]
        for c in exam_on_sametime:
            students_exam2 = set([el for s in registrations if s.registeredCourse == c.course.courseCode for el in s.studentIds])
            # print('student2',students_exam2)
            if students_exam2 :
                # print('Common', students_exam1 & students_exam2)
                if len((students_exam1 & students_exam2)) !=0:
                    hard_constraints_violations += 5
                    h1 = False

                # print(hard_constraints_violations)
      
                
        # H2: The capacity of the exam should not exceed the room capacity.
        room_capacity = 0
        for room in exam.roomNo:
            room_capacity += int(room.capacity)
        if room_capacity < len(students_exam1):
            hard_constraints_violations += 5
            h2 = False
        
        # H3 : No exam scheduled in the same room at the same time
        rooms_at_sametime = set([r.roomNumber for e in exam_on_sametime for r in e.roomNo])
        # print(rooms_at_sametime)
        # print('  kkk')
        # print([r.roomNumber for r in exam.roomNo])
        # print(rooms_at_sametime & set([r.roomNumber for r in exam.roomNo]))
        if len(rooms_at_sametime & set([r.roomNumber for r in exam.roomNo])) != 0:
            hard_constraints_violations += 5
            h3 = False
        
    # H4 : No exams are scheduled in Saturdays and Sundays
    exams_on_holidays = [e for e in chromosome if e.day[1] in ("Saturday","Sunday","Holiday")]
    if len(exams_on_holidays) !=0:
        hard_constraints_violations +=5 
        h4 = False
    
    # H5 : No nvigilator invigilating exams at the same time
    data = [(individual.invigilator, individual.day, individual.startTime) for individual in chromosome]
    for i in range(len(data)):
        for invigilator in data[i][0]:
            for j in range(len(data)):
                if i != j and invigilator in data[j][0] and data[i][1] == data[j][1] and data[i][2] == data[j][2]:
                    hard_constraints_violations += 5
                    h5 = False

        
        
        
        
    return hard_constraints_violations,h1,h2,h3,h4,h5

def calculate_soft_constraints(chromosome):
    
    soft_constraints_violations = 0
    s1 = True
    s2 = True
    s3 = True
    s4 = True
    s5 = True
    one_minus_totRooms = len(classRooms)-1 # Used in S3 
    
    # S1 Two exams in a row: minimize student sitting consecutive exams in a consecutive days.
    s1_student_count = 0
    for i in range(len(days)-1):
        if days[i] is not None and days[i+1] is not None:
            students_exam1 = set([s for e in chromosome if e.day == days[i] for r in registrations if r.registeredCourse == e.course.courseCode for s in r.studentIds])
            students_exam2 = set([s for e in chromosome if e.day == days[i+1] for r in registrations if r.registeredCourse == e.course.courseCode for s in r.studentIds])
            common_student = students_exam1 & students_exam2
            # print(len(common_student))
            s1_student_count += len(common_student)
            if len(common_student) > 10:
                # print(len(common_student))
                soft_constraints_violations += 1
                s1 = False
    # print(soft_constraints_violations)
    
    # S2 Two exams in a day: minimize student sitting more than two exams in a day.
    for day in days:
        if day is not None:
            students_exam_a_day = [r.studentIds for e in chromosome if e.day == day for r in registrations if r.registeredCourse == e.course.courseCode]
            # print(len(students_exam_a_day))
            for i in range(len(students_exam_a_day)-1):
                j = i+1
                # print(j)
                count =0
                while(j<len(students_exam_a_day)):
                    temp = set(students_exam_a_day[i]) & set(students_exam_a_day[j])
                    # print(temp)
                    j+=1
                    if len(temp)>0:
                        count+=1
                if(count >2):
                    soft_constraints_violations += 1
                    s2 = False
            
    # S3 Period penalty: minimize the number of exams scheduled in the period with a penalty.
            exam_count_per_period = 0
            for t in examStartTiming:
                exam_count_per_period = len([e for e in chromosome if e.day == day and e.startTime == t])
                # print(exam_count_per_period, one_minus_totRooms)
                if exam_count_per_period > one_minus_totRooms :
                    soft_constraints_violations += 1 
                    s3 = False
                    
     # S4 Room penalty: minimize the number of exams scheduled in a room with a penalty.
            exam_count_per_room = 0
            for r in classRooms:
                exam_count_per_room = len([e for e in chromosome if e.day == day and (r in e.roomNo)])
                # print(exam_count_per_room, totalExamStartTiming)
                if exam_count_per_room > totalExamStartTiming:
                    soft_constraints_violations += 1 
                    s4 = False
                    
    # S5 Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.   
    
    lw_exams = [e.course.courseCode for e in chromosome if e.day[2] in [14,15,16,17,18]]
    temp = set(max_count) & set(lw_exams)
    # print(temp)
    if len(temp) !=0:
        soft_constraints_violations += 1
        s5 = False
    
    
            
 
    return soft_constraints_violations,s1,s2,s3,s4,s5,s1_student_count
            
    
     
# calculate_hard_constraints(current_best_solution) 
# calculate_soft_constraints(current_best_solution) 

# Low Level Heuritsic Implementations

def random_period_assignment(current_timetable):
    # LH1: Random period assignment
    
    # Randomly select an exam from the current timetable
    selected_exam = rn.choice(current_timetable)
    
    # Get the available periods excluding the current period
    available_periods = [p for p in days if p[1] != selected_exam.day[1]]
    
    if available_periods:
        # Randomly select a new period
        new_period = rn.choice(available_periods)
        
        # Update the exam's day with the new period
        selected_exam.day = new_period
    return current_timetable
        
def random_room_assignment(current_timetable):
    # LH2: Random room assignment
    
    # Randomly select an exam from the current timetable
    selected_exam = rn.choice(current_timetable)
    total_students_in_exam = 0
    for x in registrations:
        if selected_exam.course.courseCode == x.registeredCourse:
            total_students_in_exam = len(x.studentIds)
            break
    # Get the available rooms excluding the current rooms
    available_rooms = [r for r in classRooms if r not in selected_exam.roomNo]
    
    if available_rooms:
        new_rooms = []
        
        while total_students_in_exam > 0:
            # Randomly select a new room
            new_room = rn.choice(available_rooms)
            if new_room not in new_rooms:
                new_rooms.append(new_room)
                total_students_in_exam -= int(new_room.capacity)
        
        # Update the exam's room assignment with the new room
        selected_exam.roomNo = new_rooms
    return current_timetable
        
def random_timeslot_assignment(current_timetable):
    # LH3: Random timeslot assignment
    
    # Randomly select an exam from the current timetable
    selected_exam = rn.choice(current_timetable)
    
    available_timeslots = [t for t in examStartTiming if t != selected_exam.startTime]
    
    if available_timeslots:
        
        # Randomly select a timeslot
        new_timeslot = rn.choice(available_timeslots)
        
        #Update new timeslot for the selected timeslot.
        selected_exam.startTime = new_timeslot
    return current_timetable


def apply_heuristic(i, current_solution):
    
    if i == 1:
        return random_period_assignment(current_solution)
    elif i == 2:
        return random_room_assignment(current_solution)
    elif i == 3:
        return random_timeslot_assignment(current_solution)
    
        

# current_best_solution = generate_init_solution()
# current_best_solution.sort(key= lambda x:x.day[1])
# print(evaluate(current_best_solution))

# for t in current_best_solution:
   # 1 print(t)


# iterations = 1000
    

 
# for j in range(iterations):
    
    
    # index = epsilon_decay_greedy_selection(j+1, llhList)
    # print(index)

def update_utility_sum(utility_value, flag, selected_time, discount_factor=0.9):
    
    reward_value = 1
    punish_value = -1
    
    if flag:
        utility_value += reward_value * (discount_factor ** selected_time)
    else:
        utility_value += punish_value * (discount_factor ** selected_time)
    return utility_value
    

def P(delta, temperature):
    # Your implementation for the acceptance probability function
    return random.uniform(0, 1) < min(1, pow(2.71828, -delta / temperature))


   
def simulated_annealing(total_time, T0):
    # Initialization
    current_solution = generate_init_solution()
    # for s in current_solution:
    #     print(s)
    utility_values = [0,0,0] 
    selected_times = [0,0,0]
    f_best = f_current = f_0 = quality(current_solution)
    S_best = current_solution
    start_time = time.time()
    temperature = T0
    since_imp = 0
    num_non_improving_reheat = 0
    max_reheat_times = 5
    reheat_frequency = 1000
    b_improvement_found = False

    # Main loop
    while time.time() - start_time < total_time:
        # Heuristic selection
        remaining_time = total_time - (time.time() - start_time)
        i = select_heuristic(remaining_time,utility_values)
        selected_times[i] += 1

        # Apply heuristic and evaluate
        temp_solution = apply_heuristic(i+1, current_solution)
        f_temp = quality(temp_solution)
        remaining_time = total_time - (time.time() - start_time)

        # Move acceptance
        delta = f_temp - f_current

        if delta < 0:  # Improving move
            update_utility_sum(utility_values[i], 0, True, selected_times[i])  # Update utility sum
            current_solution = temp_solution
        else:
            update_utility_sum(utility_values[i], 0, False, selected_times[i])  # Worsening move Update utility sum

        rand_value = random.uniform(0, 1)

        if delta < 0 or rand_value < P(delta, temperature):
            # Accept if non-worsening or with the Boltzmann probability
            if delta < 0:
                current_solution = temp_solution
                f_current = f_temp
                b_improvement_found = True

            if f_temp < f_best:
                S_best = temp_solution
            else:
                since_imp += 1

        if since_imp == reheat_frequency:
            # When reaching non-improvement limit
            since_imp = 0
            if not b_improvement_found:
                num_non_improving_reheat += 1
            else:
                b_improvement_found = False

            if num_non_improving_reheat > max_reheat_times:
                # When reaching max reheating limit
                num_non_improving_reheat = 0
                temperature = (T0 - temperature) / 2.0
                f_best = round(f_best * 1.1)
            else:
                temperature /= 0.85  # Increase temperature
        else:
            temperature *= 0.9  # Decrease temperature

    return S_best
s_best = simulated_annealing(300, 10000.0)
_,h1,h2,h3,h4,h5 = calculate_hard_constraints(s_best)
print("\nHard Constraints:")
print("1: An exam will be scheduled for each course\t\t\t\t ✔")
if h1:
    print("2: A student can not give more than one exam at a time\t\t\t ✔")
else:
    print("2: A student can not give more than one exam at a time\t\t\t ❌")
print("3: All exams must be held between 9 AM and 5 PM\t\t\t\t ✔")
if h2:
    print("4:The capacity of the exam should not exceed the room capacity.\t\t ✔")
else:
    print("4: The capacity of the exam should not exceed the room capacity.\t\t ❌")
if h3:
    print("5: No exam scheduled in the same room at the same time\t\t\t ✔")
else:
    print("5: No exam scheduled in the same room at the same time\t\t\t ❌")
if h4:
    print("6: No exams are scheduled in Saturdays and Sundays\t\t\t ✔")
else:
    print("6: No exams are scheduled in Saturdays and Sundays\t\t\t ❌")
if h5:
    print("7: No invigilator invigilating exams at the same time\t\t\t ✔")
else:
    print("7: No invigilator invigilating exams at the same time\t\t\t ❌")

_,s1,s2,s3,s4,s5,s1_student_count = calculate_soft_constraints(s_best)

print("\nSoft Constraints:")
print("1: An exam will be scheduled for each course\t\t\t\t ✔")
if s1:
    print("2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ✔")
else:
    print("2: Two exams in a row: minimize student sitting consecutive exams in a consecutive days.\t\t ❌", s1_student_count)
print("3: One hours of break for faculty meeting\t\t\t\t ✔")
if s2:
    print("4: Two exams in a day: minimize student sitting more than two exams in a day.\t\t ✔")
else:
    print("4: Two exams in a day: minimize student sitting more than two exams in a day.\t\t ❌")
if s3:
    print("5: Period penalty: minimize the number of exams scheduled in the period with a penalty.\t\t ✔")
else:
    print("5: Period penalty: minimize the number of exams scheduled in the period with a penalty.\t\t ❌")
if s4:
    print("6: Room penalty: minimize the number of exams scheduled in a room with a penalty.\t\t ✔")
else:
    print("6: Room penalty: minimize the number of exams scheduled in a room with a penalty.\t\t ❌")
if s5:
    print("7: Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.\t\t ✔")
else:
    print("7: Larger examination schedule late in the timetable: minimize the number of large exams that appear ‘late’ in the timetable.\t\t ❌")
quality(s_best) 
    
