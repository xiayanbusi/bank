# Here is my resume wrote by Python. If you have any questions, please feel free to contact me.#
# Contct information
Name='Feifei Han'
Position='Data Analyst'
Address="110 Erskine Ave,504"
City="Toronto"
Postal_code='M4P1Y4'
Phone="306-881-0415"
Email="hanfei0415@gmail.com"
print(Name.center(100))
print(Position.center(100))
print(Address.ljust(3))
print(City.ljust(3),Postal_code.rjust(8),sep="")
print(Phone.ljust(3))
print(Email.ljust(3))
print()
# education information
education={"Institution":["Queen's University",'George Brown College','Guilin University of Technology'],
            "Date":["2021-09 to 2022-09",'2022-09 to 2023-04','2006-09 to 2010-06'],
            "Program":["Political Studies",'Business Analysis for Decision Making',"Public Administration"],
            "Degree":["Master's"," Graduate Certificate","Bachelor's"]}
#work exprience
exprience={"Position":['ASSISTANCE MANAGER','SENIOR WEBSITE OPERATOR','WEBSITE OPERATOR'],
           'Company':['UNA PIZZERIA INC.','1932 Business Consulting Ltd.','Hainan Tianya Network technology Ltd.'],
           'Location':['Saskatoon','Saskatoon','Haikou, China'],
            'Duty':["-Monitored office inventory and kitchen supplies by tracking stock items with advanced skills in MS Excel\n-Using Excel and Tableau to visualize monthly income and expenses and generate business insights for restaurant improvements\n-Reduced 13% of kitchen food waste and 5% of dinnerware damage and saved 6.5% on human resource costs\n-Received Best Employee Award twice\n",
                    "-Content planning, data visualization, and publishing for the website and associated local newspaper\n-Led a team of 2 editors and 1 programmer to complete clients’ advertising in a fast-paced environment\n-Ranked No.1 Chinese website in Saskatoon. Developed a mobile website and increased page viewers by 35%",
                    "-Daily website management, including content censorship, database management, and web-ads design\n-Supported the sales department by generating and analyzing users’ data. Using Tableau and PowerPoint to provide real-time insights into business KPIs\n-Page viewers increased by 1271%, gained 50,000 new unique visitors, and ads sales reached 4 million RMB (0.8 million Canadian dollars)\n-Awarded Best Website Operator in 2012 among 12 sub-companies\n-Led a team of 3 full-time editors and 1 graph designer"],
           'Peirod':['2019.01-2020.03','2015.06-2018.12','2010.06-2014.05']}
#skills
Skills=['Python','R','mySQL','PowerBI','Tableau','Excel','PowerPoint']
print('Education: ')
print()
for i in range(0,3):
    print(education['Institution'][i].center(30),education['Degree'][i].rjust(45),sep="")
    print(education['Date'][i].center(30),education['Program'][i].rjust(45), sep="")
    print()
print()
print('Exprience: ')
print()
for i in range(0,3):
    print(exprience['Position'][i].ljust(15))
    print(exprience['Company'][i].ljust(10),exprience['Location'][i].rjust(15),sep="")
    print(exprience['Peirod'][i].rjust(15))
    print(exprience['Duty'][i].ljust(3))
    print()
level="*"*5
level2="*"*4
print("Skills: ")
for i in range(0,len(Skills)):
    if i in [1,2,5]:
        print(Skills[i].center(10),level2.rjust(15),sep="    ")
    else:
        print(Skills[i].center(10),level.rjust(15),sep="    ")


