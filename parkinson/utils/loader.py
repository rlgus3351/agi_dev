import customtkinter as ctk
import threading

def show_loading_overlay(parent_frame, message="불러오는 중입니다..."):
    overlay = ctk.CTkFrame(parent_frame, fg_color="gray20", corner_radius=0)  # ✅ rgba 제거
    overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

    loading_box = ctk.CTkFrame(overlay, fg_color="white", corner_radius=8)
    loading_box.place(relx=0.5, rely=0.5, anchor="center")

    ctk.CTkLabel(loading_box, text=message, text_color="black", font=ctk.CTkFont(size=14)).pack(padx=20, pady=15)

    return overlay

def run_with_loading(parent_frame, fetch_function, callback, loading_text="불러오는 중입니다..."):
    """
    ✅ CTkToplevel 대신 안전한 Frame 기반 로딩 레이어
    """
    overlay = show_loading_overlay(parent_frame, loading_text)

    def background():
        try:
            result = fetch_function()
        except Exception as e:
            result = e

        def on_done():
            overlay.destroy()  # ✅ 안전하게 제거
            callback(result)

        parent_frame.after(500, on_done)  # ⏱ 약간의 딜레이

    threading.Thread(target=background, daemon=True).start()


def show_loading_popup(parent, message="불러오는 중입니다..."):
    popup = ctk.CTkToplevel(parent)
    popup.title("로딩 중")
    popup.geometry("300x120")
    popup.transient(parent)
    popup.grab_set()
    popup.attributes("-topmost", True)
    popup.resizable(False, False)

    frame = ctk.CTkFrame(popup)
    frame.pack(expand=True, fill="both", padx=20, pady=20)

    ctk.CTkLabel(frame, text=message, font=("", 14)).pack(pady=10)

    return popup


def run_with_loading_popup(parent_frame, fetch_function, callback, loading_text="불러오는 중입니다..."):
    """
    ✅ 팝업 기반 로딩 처리 (CTkToplevel)
    """
    popup = show_loading_popup(parent_frame, loading_text)

    def background():
        try:
            result = fetch_function()
        except Exception as e:
            result = e

        def on_done():
            popup.destroy()
            callback(result)

        parent_frame.after(500, on_done)

    threading.Thread(target=background, daemon=True).start()