import sys

from ask import handle_query


if __name__ == "__main__":
    if len(sys.argv) > 1:
        one_shot_query = " ".join(sys.argv[1:]).strip()
        if one_shot_query:
            try:
                print(handle_query(one_shot_query))
            except Exception as exc:
                print(f"Error: {exc}")
        raise SystemExit(0)

    print("NYC Taxi Assistant (type 'exit' to quit)")
    while True:
        try:
            user_query = input("> ").strip()
        except EOFError:
            print("No interactive input detected. Run in a terminal or pass a query as arguments.")
            break
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue
        try:
            print(handle_query(user_query))
        except Exception as exc:
            print(f"Error: {exc}")
