"""
This script is used to download the files from Data folder in Google drive (pickle and checkpoint) into desired destination

It will store the files in the current directory- same as this file

"""


#pip install googledrivedownloader

from google_drive_downloader import GoogleDriveDownloader as gd


gd.download_file_from_google_drive(file_id='1qqxN54oUaHJlN-4-ylR8IPQRcU1B7yWa',dest_path='./vgg16-20160129.tfmodel', unzip=True)
gd.download_file_from_google_drive(file_id='19cNqkcmnWS3zyZrLW4rXRlTf7WH7IfbL',dest_path='./vocab_file2.pkl', unzip=True)
gd.download_file_from_google_drive(file_id='1n3FdwQ7eEiHfmrQ-hB-rEwhkJueJz1LB',dest_path='./val_image_id_list.h5', unzip=True)
gd.download_file_from_google_drive(file_id='1zL5_LA7dlyPMsm8VN7TKtKYh00RVHrqd',dest_path='./val_fc7.h5', unzip=True)
gd.download_file_from_google_drive(file_id='1p4wPTgOzeB3C-PyQ5ZfjuYlY-BKgMetg',dest_path='./train_image_id_list.h5', unzip=True)
gd.download_file_from_google_drive(file_id='10LtxVTPshL0IUBTgZ-Pd9RUx_9eHILkl',dest_path='./train_fc7.h5', unzip=True)
gd.download_file_from_google_drive(file_id='1pYyukqLM7OxKA-eS5sYXbdRQH4Wc2Zy1',dest_path='./qa_data_file2.pkl', unzip=True)
gd.download_file_from_google_drive(file_id='1koPE3rhwuml8E2hSOXO7-8ry7k7T34ov',dest_path='./model59.ckpt.meta', unzip=True)
gd.download_file_from_google_drive(file_id='1lQYKQh907JveQOdIKCNxQRenCJyJruND',dest_path='./model59.ckpt.index', unzip=True)
gd.download_file_from_google_drive(file_id='17q1H9L90aa4W91111OLdaBonzQWTz18I',dest_path='./model59.ckpt.data-00000-of-00001', unzip=True)
gd.download_file_from_google_drive(file_id='1DLF25bGVY-9fGmzpmj5QQhk7m_mrgyuj',dest_path='./checkpoint', unzip=True)

"""
ACTUAL GOOGLE DRIVE LINKS:
https://drive.google.com/open?id=1qqxN54oUaHJlN-4-ylR8IPQRcU1B7yWa
https://drive.google.com/open?id=19cNqkcmnWS3zyZrLW4rXRlTf7WH7IfbL
https://drive.google.com/open?id=1n3FdwQ7eEiHfmrQ-hB-rEwhkJueJz1LB
https://drive.google.com/open?id=1zL5_LA7dlyPMsm8VN7TKtKYh00RVHrqd
https://drive.google.com/open?id=1p4wPTgOzeB3C-PyQ5ZfjuYlY-BKgMetg
https://drive.google.com/open?id=10LtxVTPshL0IUBTgZ-Pd9RUx_9eHILkl
https://drive.google.com/open?id=1pYyukqLM7OxKA-eS5sYXbdRQH4Wc2Zy1
https://drive.google.com/open?id=1koPE3rhwuml8E2hSOXO7-8ry7k7T34ov
https://drive.google.com/open?id=1lQYKQh907JveQOdIKCNxQRenCJyJruND
https://drive.google.com/open?id=17q1H9L90aa4W91111OLdaBonzQWTz18I
https://drive.google.com/open?id=1DLF25bGVY-9fGmzpmj5QQhk7m_mrgyuj
"""