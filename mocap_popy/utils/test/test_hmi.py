import mocap_popy.utils.hmi as hmi

if __name__ == "__main__":
    ui = hmi.get_user_input("Test no choices")
    print(ui)

    ui = hmi.get_user_input(
        "Test with choices", choices=[hmi.YES_KEYWORDS, hmi.NO_KEYWORDS]
    )
    print(ui)

    ui = hmi.get_user_input("Test with exit on quit", exit_on_quit=True)
    print(ui)
