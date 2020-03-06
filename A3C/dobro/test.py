import threading
import time

def main():
    global a
    num_thread = 3
    a = 0

    def func(t_idx):
        global a

        while a < 10:
            a += 1
            print(t_idx, a)
            time.sleep(0.1)

    threads = []
    for i in range(num_thread):
        threads.append(threading.Thread(target=func, args=(i+1,)))
        threads[-1].start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()