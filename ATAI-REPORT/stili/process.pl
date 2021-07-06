#!/usr/bin/perl
# licensed under the GPL, by Lars Madsen, 2008/03/25
use Getopt::Long;
my $f = '';
my $k = '';
my $p = '';
my $tmppdf  = 'tmp.pdf';
my $postfix = '-style';
GetOptions('f:s' => \$f,'k:s' => \$k,'p:s' => \$p);
my @styles = ();
my %pages  = ();
print <<END if ! $f;
Usage:

   $0 -f MemoirChapStyles.styles

END
exit if ! $f;

if ( $k ) { compile_file("$k$postfix"); exit ;}
open my $file ,'<', $f or die "Cannot open '$f': $!";
for my $l (<$file>) {
    chomp $l;
    next if $l =~ /^\s*$/;
    if ( $l =~ / page/ ) {
 ($Page) = ( $l =~ / page (.*)/ ) ;
 $l =~ s/ page.*//;
 $pages{$l} = $Page;
    }
    push @styles,$l;
}
close $file;
for my $style ( @styles ) {
    compile_file($style);
}
print "done\n\n";
sub compile_file {
    my $style = shift;
    my @tmp = ();
    system("pdflatex", "$style.tex")          == 0 or warn "$!";
    system("pdfcrop", "$style.pdf","$tmppdf") == 0 or warn "$!";
    system("mv", "$tmppdf","$style.pdf")      == 0 or warn "$!";
    if ( $pages{$style} || $p ) {
 @tmp = split /\,/,  $pages{$style} ? $pages{$style} : $p ;
 for my $p ( @tmp ) {
     system("pdftops", "-eps","-f","$p","-l","$p", "$style.pdf", "$style-$p.eps" )   == 0 or warn "$!";
     warn "Created $style-$p.eps\n";
 }
    }
    else {
 system("pdftops", "-eps", "$style.pdf")   == 0 or warn "$!";
    }
    print "Done converting $style.pdf\n";
    return;
}
