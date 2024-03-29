# Project Management using Gantt Chart

Tracking schedules and tasks for Project Management is made possible by the Gantt Chart. Althought this template is created in Excel it very much resembles the one available 
in the Microsoft Project and is a very effective template for Project Management tasks. 

Features of the Gantt Chart:
- Date is marked by a blue vertical line
- Changes dynamically when you change the start date
- Colored boxes depending you the status (Not Started, In Progress, Blocked, Complete) of the project
- Completed tasks are represented by the golden color
- In progress tasks are represented by the blue  color
- Diamond shape represents the end date of the project
- Blocked tasks are represented by orange tasks

![Gantt Chart](https://user-images.githubusercontent.com/62851341/210553235-c5148517-a289-4245-b9fd-6b05cddfbaa7.png)

Access the template [here](https://docs.google.com/spreadsheets/d/1JFD-41rFEbElcw7yG2QJsYm8WCchVtgJ/edit?usp=share_link&ouid=104973142209078855674&rtpof=true&sd=true).

Below are some of the functions used to improve the dynamics of the template:
1. Function used to create the Calender:
=IF(MONTH(D3-WEEKDAY((D3),2)+1)<MONTH(D3), (D3-28-DAY(D3)+7)- WEEKDAY((D3-DAY(D3)+7),2)+1, (D3-DAY(D3)+7)- WEEKDAY((D3-DAY(D3)+7),2)+1)

2. Function used to create the "Number of working days" column:
=IF(F6="","",NETWORKDAYS(E6,F6))

3. Function used to create Diamond shapes to represent end date:
=IF(K$4=($F6-WEEKDAY($F6,2)),"u","")

4. Function used to create the blue line to represent today's date:
=K$4=(TODAY()-WEEKDAY(TODAY(), 2) + 1)

5. Function used to create the colored boxes representing the start and end date:
=AND(K$4>=$E6-(WEEKDAY($E6, 2)+1),K$4<=$F6)

6. Function used to show percentage complete:
=AND($I6>0, K$4 <= ($E6+($F6-$E6)×$I6)-WEEKDAY(($E6+($F6-$E6)×$I6),2)+1, K$4 >= $E6-WEEKDAY($E6, 2)+1)

7. Function used to create the to show completed tasks using golden color:
=AND($H6="Complete", K$4 = $F6 - WEEKDAY($F6,2) + 1)

8. Function used to create the to show blocked tasks using orange color:
=AND($H6="Blocked",$I6>0,K$4<=($E6+($F6-$E6)×$I6)-WEEKDAY(($E6+($F6-$E6)×$I6),2)+1,K$4>=$E6-WEEKDAY($E6,2)+1)
