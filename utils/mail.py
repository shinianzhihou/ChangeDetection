import smtplib
import sys
from email.header import Header
from email.mime.text import MIMEText
 
# 第三方 SMTP 服务
mail_host = "smtp.qq.com"      # SMTP服务器
mail_user = "1178396201@qq.com"                  # 用户名
mail_pass = "mgkjoyidyadfhhhg"               # 授权密码，非登录密码
 
sender = "1178396201@qq.com"    # 发件人邮箱(最好写全, 不然会失败)
 
def sendEmail():
 
    message = MIMEText(content, 'plain', 'utf-8')  # 内容, 格式, 编码
    message['From'] = "{}".format(sender)
    message['To'] = receivers
    message['Subject'] = title
 
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)  # 启用SSL发信, 端口一般是465
        smtpObj.login(mail_user, mail_pass)  # 登录验证
        smtpObj.sendmail(sender, receivers, message.as_string())  # 发送
        print("mail has been send successfully.")
    except smtplib.SMTPException as e:
        print(e)
 
if __name__ == '__main__':
    
    receivers = sys.argv[1]  #收件人邮箱
    title = sys.argv[2]      #标题
    content = sys.argv[3]    #内容
    sendEmail()