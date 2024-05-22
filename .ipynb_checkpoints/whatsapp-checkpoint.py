import time
import pyautogui
import pygetwindow as gw

# Function to send audio file to WhatsApp group
def send_audio_to_whatsapp(audio_path):
    # Open WhatsApp Desktop application
    pyautogui.press('win')
    time.sleep(1)
    pyautogui.write('WhatsApp', interval=0.1)
    time.sleep(1)
    pyautogui.press('enter')
    time.sleep(10)

    # Click on the search bar
    pyautogui.click(x=221, y=221)  # Adjust coordinates as per your screen resolution
    time.sleep(1)

    # Type the name of the group/contact
    pyautogui.write('TRIVANDRUM PM2.5 ALERT!', interval=0.1)
    time.sleep(5)

    # Click on the chat
    pyautogui.click(x=311, y=328)  # Adjust coordinates as per your screen resolution
    time.sleep(1)

    # Click on the attachment icon
    pyautogui.click(x=854, y=954)  # Adjust coordinates as per your screen resolution
    time.sleep(1)

    # Ensure the WhatsApp window is in focus
    whatsapp_win = gw.getWindowsWithTitle('WhatsApp')[0]
    whatsapp_win.activate()

    # Double click on the "Document" option
    document_location = (734, 715)  # Adjust coordinates as per your screen resolution
    pyautogui.moveTo(document_location)
    pyautogui.doubleClick()
    time.sleep(1)

    # Type the path of the audio file and press 'Enter'

    pyautogui.write(audio_path, interval=0.1)
    time.sleep(1)
    pyautogui.press('enter')
    time.sleep(3)
    pyautogui.click(x=1672, y=940)

# Example usage
audio_path = r"C:\Users\Gatha Reghunath\Desktop\voice\templates\peak_pm25.mp3"
send_audio_to_whatsapp(audio_path)
