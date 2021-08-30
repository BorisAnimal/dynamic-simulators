from multiprocessing import Value


class AsyncPlant:
    """
    Общий интерфейс для параллельных симуляторов
    """

    def __init__(self):
        self.t = None
        self.control_hz = None
        self.u = Value('d', 0.0)
        self.process = None
        self.state = None

    def get_state(self):
        """
        Method to get current state of system
        :return: state variables i.e. (state, t)
        """
        pass

    def set_control(self, u):
        """
        Setup current control value to actuator
        :param u: double
        """
        pass

    def start(self, **kwargs):
        """
        Единственная точка входа в начало эксперимента
        """
        pass

    def terminate(self):
        """
        Единственный метод принудительного прекращения эксперимента
        """
        pass

    def is_running(self):
        """
        Проверка активен ли процесс эксперимента
        :return: boolean
        """
        pass

    def on_success(self):
        """
        Метод после успешного завершения эксперимента - сохранение данных и т.п.
        """
        pass

    def save_history(self):
        pass

    def load_history(self):
        pass
