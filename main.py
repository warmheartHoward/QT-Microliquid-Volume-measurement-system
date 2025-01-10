from PyQt5.QtWidgets import QApplication
from login_window import LoginWindow
from main_window import MainWindow
import sys

def main():
    app = QApplication(sys.argv)
    
    # Create and show login window
    login_window = LoginWindow()
    login_window.show()
    
    # Create main window (hidden initially)
    main_window = None
    
    def show_main_window():
        nonlocal main_window
        username = login_window.username_input.text()
        main_window = MainWindow(username)
        main_window.show()
    
    # Connect login success signal to show main window
    login_window.login_success.connect(show_main_window)
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
