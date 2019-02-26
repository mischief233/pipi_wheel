#coding=utf-8
#功能：读取数据，根据关键字匹配情况赋值1或0，将数据写入新的文件当中
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import xlrd
import xlwt
#读文件
data=xlrd.open_workbook('kouhong.xlsx')
table=data.sheets()[0]#0开始，选第一个sheet
x=table.col_values(19)#0开始，选取列信息，第20列
#写文件初始化
workbook = xlwt.Workbook()
sheet1 = workbook.add_sheet('sheet1',cell_overwrite_ok=True)#创建sheet
j=0
for element in x[1:]:
    if '滋润' in element or '保湿' in element or '补水' in element:
        sheet1.write(j,0,1)#(row ,col ,value)
    else:
        sheet1.write(j,0,0)
        
    if '脱色' in element or '持久' in element or '不掉色' in element or '不脱装' in element:
        sheet1.write(j,1,1)
    else:
        sheet1.write(j,1,0)
        
    if '易' in element:
        sheet1.write(j,2,1)
    else:
        sheet1.write(j,2,0)
    j=j+1
workbook.save('./note.xls')#保存新文件

