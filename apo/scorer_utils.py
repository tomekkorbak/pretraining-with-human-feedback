import re
import logging
from dateparser.search import search_dates
from datetime import datetime
from typing import List, Generator, Dict, Optional

from scrubadub.filth.base import Filth
from scrubadub.filth.date_of_birth import DateOfBirthFilth
try:
    import pyap
    # import postal.parser
except (ImportError, ):
    raise ImportError(
        'To use scrubadub_address.detectors.address, extra dependencies need to be installed: pyap and postal. '
        'See https://scrubadub.readthedocs.io/en/stable/addresses.html for more details on how to install these.'
    )
from scrubadub.detectors.catalogue import register_detector
from scrubadub.detectors.base import Detector
from scrubadub.detectors.postalcode import PostalCodeDetector
from scrubadub.filth.address import AddressFilth

# if pkg_resources.get_distribution('pyap').version.split('.') '0.3.1':
# A little monkey patching to fix the postcode regex
import pyap.source_GB.data
pyap.source_GB.data.full_address = r"""
    (?P<full_address>
        {full_street}
        (?: {part_divider} {city} )?
        (?: {part_divider} {region1} )?
        {part_divider}? {postal_code}
        (?: {part_divider} {country} )?
    )  # end full_address
""".format(
    full_street=pyap.source_GB.data.full_street,
    part_divider=pyap.source_GB.data.part_divider,
    city=pyap.source_GB.data.city,
    region1=pyap.source_GB.data.region1,
    country=pyap.source_GB.data.country,
    postal_code="(?P<postal_code>" + PostalCodeDetector.region_regex['GB'].pattern + ")",
)


@register_detector
class AddressDetectorNoLibpostal(Detector):
    """This ``Detector`` aims to detect addresses.

    This detector uses some complex dependencies and so is not enabled by default. To install the needed python
    dependencies run:

    .. code-block:: bash

        pip install scrubadub[address]

    This detector is based on the python package `pyap <https://pypi.org/project/pyap/>`_ and so only supports the
    countries that pyap supports: US, GB and CA. The results from `pyap` are cross-checked using
    `pypostal <https://github.com/openvenues/pypostal>`_, which builds upon openvenues'
    `libpostal <https://github.com/openvenues/libpostal>`_ library. libpostal needs to be compiled from source and
    instructions can be found on on their github `<https://github.com/openvenues/libpostal>`_

    After installing the python dependencies and libpostal, you can use this detector like so:

    >>> import scrubadub, scrubadub_address
    >>> scrubber = scrubadub.Scrubber()
    >>> scrubber.add_detector(scrubadub_address.detectors.AddressDetector)
    >>> scrubber.clean("I live at 6919 Bell Drives, East Jessicastad, MO 76908")
    'I live at {{ADDRESS}}'

    """
    filth_cls = AddressFilth
    name = 'address'
    ignored_words = ["COVERAGE"]

    def __init__(self, *args, **kwargs):
        """Initialise the ``Detector``.

        :param name: Overrides the default name of the :class:``Detector``
        :type name: str, optional
        :param locale: The locale of the documents in the format: 2 letter lower-case language code followed by an
                       underscore and the two letter upper-case country code, eg "en_GB" or "de_CH".
        :type locale: str, optional
        """
        super(AddressDetectorNoLibpostal, self).__init__(*args, **kwargs)

        self.match_pyap_postal_fields = {}  # type: Dict[str, str]
        self.minimum_address_sections = 0
        if self.region == 'US':
            self.match_pyap_postal_fields = {'region1': 'state'}
            self.minimum_address_sections = 4

    @classmethod
    def supported_locale(cls, locale: str) -> bool:
        """Returns true if this ``Detector`` supports the given locale.

        :param locale: The locale of the documents in the format: 2 letter lower-case language code followed by an
                       underscore and the two letter upper-case country code, eg "en_GB" or "de_CH".
        :type locale: str
        :return: ``True`` if the locale is supported, otherwise ``False``
        :rtype: bool
        """
        language, region = cls.locale_split(locale)
        return region in ['GB', 'CA', 'US']

    def iter_filth(self, text, document_name: Optional[str] = None):
        """Yields discovered filth in the provided ``text``.

        :param text: The dirty text to clean.
        :type text: str
        :param document_name: The name of the document to clean.
        :type document_name: str, optional
        :return: An iterator to the discovered :class:`Filth`
        :rtype: Iterator[:class:`Filth`]
        """
        addresses = pyap.parse(text, country=self.region)
        for address in addresses:
            # Ignore any addresses containing any explitally ignored words
            if any([word.lower() in address.full_address.lower() for word in self.ignored_words]):
                # print("contains an ignored word")
                continue

            # postal_address = None
            # if self.minimum_address_sections > 0:
            #     postal_address = postal.parser.parse_address(address.full_address)
            #     # Ensure that there are enough parts of the address to be a real address
            #     if len(postal_address) < self.minimum_address_sections:
            #         # print("address too short")
            #         continue

            # if len(self.match_pyap_postal_fields) > 0:
            #     if postal_address is None:
            #         postal_address = postal.parser.parse_address(address.full_address)
            #     # Check the two parses agree on part of the address
            #     for pyap_field, postal_field in self.match_pyap_postal_fields.items():
            #         if not address.__getattribute__(pyap_field).lower() in [
            #             part[0] for part in postal_address if part[1] == postal_field
            #         ]:
            #             continue

            # It seems to be a real address, lets look for it in the text
            # This is needed as pyap does some text normalisation, this undoes that normalisation
            # See _normalize_string() in https://github.com/vladimarius/pyap/blob/master/pyap/parser.py
            pattern = re.escape(address.full_address)
            # in python3.6 re.escape escapes ',' as '\,', later versions do not.
            # The first pattern.replace is for the earlier python versions, while the second one is for the
            # newer versions of python
            pattern = pattern.replace('\\,\\ ', '\\s*([\\n,]\\s*)+')
            pattern = pattern.replace(',\\ ', '\\s*([\\n,]\\s*)+')
            pattern = pattern.replace(r'\ ', r'\s+')
            pattern = pattern.replace('-', '[‐‑‒–—―]')
            pattern = r'\b' + pattern + r'\b'
            found_strings = re.finditer(pattern, text, re.MULTILINE | re.UNICODE)

            # Iterate over each found string matching this regex and yield some filth
            for instance in found_strings:
                yield self.filth_cls(
                    beg=instance.start(),
                    end=instance.end(),
                    text=instance.group(),
                    detector_name=self.name,
                    document_name=document_name,
                    locale=self.locale,
                )


@register_detector
class DateOfBirthDetectorNonNan(Detector):
    """This detector aims to detect dates of birth in text.

    First all possible dates are found, then they are filtered to those that would result in people being between
    ``DateOfBirthFilth.min_age_years`` and ``DateOfBirthFilth.max_age_years``, which default to 18 and 100
    respectively.

    If ``require_context`` is True, we search for one of the possible ``context_words`` near the found date. We search
    up to ``context_before`` lines before the date and up to ``context_after`` lines after the date. The context that
    we search for are terms like `'birth'` or `'DoB'` to increase the likelihood that the date is indeed a date of
    birth. The context words can be set using the ``context_words`` parameter, which expects a list of strings.

    >>> import scrubadub, scrubadub.detectors.date_of_birth
    >>> DateOfBirthFilth.min_age_years = 12
    >>> scrubber = scrubadub.Scrubber(detector_list=[
    ...     scrubadub.detectors.date_of_birth.DateOfBirthDetector(),
    ... ])
    >>> scrubber.clean("I was born on 10-Nov-2008.")
    'I was born {{DATE_OF_BIRTH}}.'

    """
    name = 'date_of_birth'
    filth_cls = DateOfBirthFilth
    autoload = False

    context_words_language_map = {
        'en': ['birth', 'born', 'dob', 'd.o.b.'],
        'de': ['geburt', 'geboren', 'geb', 'geb.'],
    }

    def __init__(self, context_before: int = 2, context_after: int = 1, require_context: bool = True,
                 context_words: Optional[List[str]] = None, **kwargs):
        """Initialise the detector.

        :param context_before: The number of lines of context to search before the date
        :type context_before: int
        :param context_after: The number of lines of context to search after the date
        :type context_after: int
        :param require_context: Set to False if your dates of birth are not near words that provide context (such as
            "birth" or "DOB").
        :type require_context: bool
        :param context_words: A list of words that provide context related to dates of birth, such as the following:
            'birth', 'born', 'dob' or 'd.o.b.'.
        :type context_words: bool
        :param name: Overrides the default name of the :class:``Detector``
        :type name: str, optional
        :param locale: The locale of the documents in the format: 2 letter lower-case language code followed by an
                       underscore and the two letter upper-case country code, eg "en_GB" or "de_CH".
        :type locale: str, optional
        """
        super(DateOfBirthDetectorNonNan, self).__init__(**kwargs)

        self.context_before = context_before
        self.context_after = context_after
        self.require_context = require_context

        try:
            self.context_words = self.context_words_language_map[self.language]
        except KeyError:
            raise ValueError("DateOfBirthDetector does not support language {}.".format(self.language))

        if context_words is not None:
            self.context_words = context_words

        self.context_words = [word.lower() for word in self.context_words]

    def iter_filth(self, text: str, document_name: Optional[str] = None) -> Generator[Filth, None, None]:
        """Search ``text`` for ``Filth`` and return a generator of ``Filth`` objects.

        :param text: The dirty text that this Detector should search
        :type text: str
        :param document_name: Name of the document this is being passed to this detector
        :type document_name: Optional[str]
        :return: The found Filth in the text
        :rtype: Generator[Filth]
        """

        # using the dateparser lib - locale can be set here
        try:
            date_picker = search_dates(text, languages=[self.language])
        except RecursionError:
            logger = logging.getLogger("scrubadub.detectors.date_of_birth.DateOfBirthDetector")
            logger.error(f"The document '{document_name}' caused a recursion error in dateparser.")
            raise
        if date_picker is None:
            return None

        lines = text.split('\n')

        for identified_string, identified_date in date_picker:
            # Skip anything that could be a phone number, dates rarely begin with a plus
            suspected_phone_number = str(identified_string).startswith('+')
            if suspected_phone_number:
                continue

            # Skip any dates that fall outside of the configured age range
            years_since_identified_date = datetime.now().year - identified_date.year
            within_age_range = (DateOfBirthFilth.min_age_years <= years_since_identified_date <=
                                DateOfBirthFilth.max_age_years)
            if not within_age_range:
                continue

            # If its desired, search for context, if no context is found skip this identified date
            if self.require_context:
                found_context = False
                # Search line by line for the identified date string (identified_string)
                for i_line, line in enumerate(lines):
                    if identified_string not in line:
                        continue
                    # when you find the identified_string, search for context
                    from_line = max(i_line - self.context_before, 0)
                    to_line = max(i_line + self.context_after + 1, 0)
                    text_context = ' '.join(lines[from_line:to_line]).lower()
                    found_context = any(context_word in text_context for context_word in self.context_words)
                    # If you find any context around any instances of this string, all instance are PII
                    if found_context:
                        break
                # If we didn't find any context, this isnt PII, so skip this date
                if not found_context:
                    continue

            found_dates = re.finditer(re.escape(identified_string), text)

            for instance in found_dates:
                begin = instance.start()
                endin = instance.end()
                if (begin is None) or (endin is None) or (begin >= endin): continue
                yield DateOfBirthFilth(
                    beg=begin,
                    end=endin,
                    text=instance.group(),
                    detector_name=self.name,
                    document_name=document_name,
                    locale=self.locale,
                )

    @classmethod
    def supported_locale(cls, locale: str) -> bool:
        """Returns true if this ``Detector`` supports the given locale.

        :param locale: The locale of the documents in the format: 2 letter lower-case language code eg "en", "es".
        :type locale: str
        :return: ``True`` if the locale is supported, otherwise ``False``
        :rtype: bool
        """
        language, region = cls.locale_split(locale)
        return language in cls.context_words_language_map.keys()
