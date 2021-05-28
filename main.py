import telebot
import config
import photo_conversion
import keras_model


def report(src, message):
    # Ответ на изображение пользователя.
    # Отправляет в сообщении скор, высчитанный моделью из файла keras_model.py
    img = photo_conversion.convert_photo(src, 224)

    result = keras_model.predict(model, img)
    good = '{:.3f}'.format(100 * result[0][0])
    bad = '{:.3f}'.format(100 * result[0][1])
    answer = 'Заключение нейросети: \n' + "Доброкачественная на " + good + "%\nЗлокачественная на " + bad + "%"
    bot.send_message(message.chat.id, answer)


if __name__ == '__main__':
    bot = telebot.TeleBot(config.TOKEN)
    model = keras_model.init_model()    # инициализируем модель при запуске бота


    @bot.message_handler(content_types=['text'])
    def lala(message):
        bot.send_message(message.chat.id, 'Добрый день! Это учебный бот, демонстрирующий работу нейросети, распознающей'
                                          ' рак кожи. Пришлите фотографию родинки, чтобы проверить функционирование')


    @bot.message_handler(content_types=['photo'])
    # обработка сжатых фотографий
    def photo_answer(message):
        src = message.from_user.username + '.png'
        file_info = bot.get_file(message.photo[0].file_id)
        downloaded_photo = bot.download_file(file_info.file_path)
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_photo)
            new_file.close()
        bot.send_message(message.chat.id,
                         'Ваше сообщение успешно отправлено нашим индийским коллегам, иммитирующим нейросеть. '
                         'Результат будет доступен через некоторое время.')
        bot.send_chat_action(message.chat.id, 'typing')
        report(src, message)


    @bot.message_handler(content_types=['document'])
    # обработка несжатых фотографий
    def doc_answer(message):
        src = message.from_user.username + '.jpg'
        try:
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

            bot.reply_to(message, 'Ваше сообщение успешно отправлено нашим индийским коллегам, иммитирующим нейросеть. '
                                  'Результат будет доступен через некоторое время.')
        except Exception as e:
            bot.reply_to(message, e)

        report(src, message)


bot.polling(none_stop=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
