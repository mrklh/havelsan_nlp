NLTK_STOP_WORDS = [
    'acaba',
    'ama',
    'aslında',
    'az',
    'bazı',
    'belki',
    'biri',
    'birkaç',
    'birşey',
    'biz',
    'bu',
    'da',
    'daha',
    'de',
    'defa',
    'diye',
    'en',
    'eğer',
    'gibi',
    'hem',
    'hep',
    'hepsi',
    'her',
    'hiç',
    'ile',
    'ise',
    'için',
    'kez',
    'ki',
    'kim',
    'mu',
    'mü',
    'mı',
    'nasıl',
    'ne',
    'neden',
    'nerde',
    'nerede',
    'nereye',
    'niye',
    'niçin',
    'o',
    'sanki',
    'siz',
    'tüm',
    've',
    'veya',
    'ya',
    'yani',
    'çok',
    'çünkü',
    'şey',
    'şu'
]

AVAILABLE_STOP_WORDS = ['number', 'bir', 'göre', 'sonra', 'kadar', 'acaba', 'altmış', 'altı', 'ama', 'ancak', 'arada',
                        'aslında', 'ayrıca', 'bana', 'bazı', 'belki', 'ben', 'benden', 'beni', 'benim', 'beri', 'beş',
                        'bile', 'bin', 'bir', 'birçok', 'biri', 'birkaç', 'birkez', 'birşey', 'birşeyi', 'biz', 'bize',
                        'bizden', 'bizi', 'bizim', 'böyle', 'böylece', 'bu', 'buna', 'bunda', 'bundan', 'bunlar',
                        'bunları',
                        'bunların', 'bunu', 'bunun', 'burada', 'çok', 'çünkü', 'da', 'daha', 'dahi', 'de', 'defa',
                        'değil',
                        'diğer', 'diye', 'doksan', 'dokuz', 'dolayı', 'dolayısıyla', 'dört', 'edecek', 'eden', 'ederek',
                        'edilecek', 'ediliyor', 'edilmesi', 'ediyor', 'eğer', 'elli', 'en', 'etmesi', 'etti', 'ettiği',
                        'ettiğini', 'gibi', 'göre', 'halen', 'hangi', 'hatta', 'hem', 'henüz', 'hep', 'hepsi', 'her',
                        'herhangi', 'herkesin', 'hiç', 'hiçbir', 'için', 'iki', 'ile', 'ilgili', 'ise', 'işte',
                        'itibaren',
                        'itibariyle', 'kadar', 'karşın', 'katrilyon', 'kendi', 'kendilerine', 'kendini', 'kendisi',
                        'kendisine', 'kendisini', 'kez', 'ki', 'kim', 'kimden', 'kime', 'kimi', 'kimse', 'kırk',
                        'mu', 'mü', 'mı', 'nasıl', 'ne', 'neden', 'nedenle', 'nerde', 'nerede', 'nereye', 'niye',
                        'niçin', 'o', 'olan', 'olarak', 'oldu', 'olduğu', 'olduğunu', 'olduklarını', 'olmadı',
                        'olmadığı',
                        'olmak', 'olması', 'olmayan', 'olmaz', 'olsa', 'olsun', 'olup', 'olur', 'olursa', 'oluyor',
                        'on',
                        'ona', 'ondan', 'onlar', 'onlardan', 'onları', 'onların', 'onu', 'onun', 'otuz', 'oysa', 'öyle',
                        'pek', 'rağmen', 'sadece', 'sanki', 'sekiz', 'seksen', 'sen', 'senden', 'seni', 'senin', 'siz',
                        'sizden', 'sizi', 'sizin', 'şey', 'şeyden', 'şeyi', 'şeyler', 'şöyle', 'şu', 'şuna', 'şunda',
                        'şundan', 'şunları', 'şunu', 'tarafından', 'tüm', 'üç', 'üzere', 'var', 'vardı',
                        've', 'veya', 'ya', 'yani', 'yapacak', 'yapılan', 'yapılması', 'yapıyor', 'yapmak', 'yaptı',
                        'yaptığı', 'yaptığını', 'yaptıkları', 'yedi', 'yerine', 'yetmiş', 'yine', 'yirmi', 'yoksa',
                        'yüz',
                        'zaten', 'altmış', 'altı', 'bazı', 'beş', 'birşey', 'birşeyi', 'mi', 'kırk', 'mı', 'nasıl',
                        'onlari', 'onların', 'yetmiş', 'şey', 'şeyden', 'şeyi', 'şeyler', 'şu', 'şuna', 'şunda',
                        'şundan',
                        'şunu', 'un']

BLACK_LIST = ['olarak', 'olan', 'kadar', 'yeni', 'yüzde', 'sonra', 'göre', 'büyük', 'son',
              'türkiye', 'olduğunu', 'iyi', 'ancak', 'olduğu', 'bin', 'önemli', 'yer', 'yıl',
              'nin', 'ın', 'önce', 'fazla', 'değil', 'dedi', 'in', 'iki', 'devam',
              'arasında', 'ilgili', 'aynı', 'oldu', 'zaman', 'tarafından', 'sadece', 'yapılan',
              'içinde', 'gün', 'birlikte', 'şekilde', 'sahip', 'nın', 'diğer', 'eden', 'geçen',
              'özel', 'söyledi', 'ortaya', 'yok', 'ben', 'özellikle', 'teknik',
              'farklı', 'gelen', 'etti', 'ardından', 'kendi', 'dr', 'alan', 'bunun', 'başkanı',
              'karşı', 'olacak', 'uzun', 'yaptığı', 'tek', 'nedeniyle', 'olması', 'bile',
              'ifade', 'dikkat', 'olmak', 'üzerinde', 'konuştu', 'ayrıca', 'bine', 'den', 'milyon']

APRIORI_LIST = apriori_list = {
    'economics':
        ['milyar', 'tipi', 'dolar', 'para', 'tl',
         'fon', 'lira', 'oranı',
         'bankası', 'genel', 'yatırım', 'yılında', 'iş', 'kredi',
         'eğitim', 'ye', 'stanbul', 'toplam', 'zarar', 'kurulu', 'faiz', 'ayında', 'kâr',
         'yılın', 'çeyrek', 'merkez', 'ücretsiz', 'fonu', 'den', 'yönetim',
         'forex', 'rehberi', 'yılı çeyrek', 'kâr zarar', 'sanal para', 'öğrenin'],
    'health':
        ['tedavi', 'su', 'sağlık', 'sık', 'sağlıklı', 'prof', 'kalp',
         'olabilir', 'yol', 'kan',
         'kilo', 'yağ', 'diş', 'çocuk', 'beslenme', 'göz', 'bağlı',
         'uygun', 'erken', 'cinsel', 'aşırı', 'tercih', 'nedenle', 'cilt', 'uzmanı', 'yıl',
         'doğum', 'doğal', 'birçok', 'yaş', 'gerekir', 'bebek', 'yerine', 'saç', 'mutlaka',
         'yaşam', 'kanser', 'kişinin', 'bile'],
    'life':
        ['istanbul',
         'böyle', 'kadın', 'benim', 'başka', 'güzel',
         'üzerine', 'doğru', 'film', 'yine',
         'anne', 'küçük', 'ünlü', 'biraz', 'şöyle', 'bana', 'genç', 'pek', 'eski',
         'öyle', 'dan', 'üç', 'tam', 'saat', 'şimdi', 'onun', 'dünya', 'kişi',
         'hemen'],
    'sports':
        ['fenerbahçe', 'galatasaray', 'teknik', 'beşiktaş', 'takım', 'sezon',
         'gol', 'maç', 'futbol', 'spor', 'transfer', 'maçta', 'maçında',
         'maçı', 'oyuncu', 'milli', 'lig', 'trabzonspor', 'futbolcu', 'süper', 'dk',
         'dakikada', 'direktör', 'avrupa', 'teknik direktör',
         'başkan', 'takımın', 'ligde', 'ikinci', 'ligi', 'hafta', 'forma',
         'yaptı', 'maçın', 'takımı', 'uefa', 'kulübü', 'türk',
         'ceza'],
    'technology':
        ['tıklayın', 'iphone', 'google', 'internet', 'dünyanın',
         'akıllı', 'apple', 'oyun',
         'windows', 'çalışma', 'imkanı', 'telefon', 'mobil',
         'sistemi', 'araç', 'facebook', 'sayesinde', 'üzerinden', 'ön',
         'sanal', 'dünyanın ünlü', 'şirketlerinde', 'çalışma imkanı', 'şirketlerinde çalışma',
         'imkanı tıklayın', 'ünlü şirketlerinde çalışma', 'dünyanın ünlü şirketlerinde',
         'çalışma imkanı tıklayın', 'ünlü şirketlerinde', 'şirketlerinde çalışma imkanı',
         'android', 'cep', 'teknoloji']
}